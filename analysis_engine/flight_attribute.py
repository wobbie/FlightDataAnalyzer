# -*- coding: utf-8 -*-
##############################################################################


import numpy as np
import pytz

from collections import Counter
from datetime import datetime
from operator import itemgetter

from analysis_engine import __version__, settings
from analysis_engine.api_handler import get_api_handler, NotFoundError

from analysis_engine.library import (
    all_of,
    any_of,
    datetime_of_index,
    min_value,
    max_value,
    most_common_value,
)
from analysis_engine.node import A, KTI, KPV, FlightAttributeNode, M, P, S


##############################################################################
# Superclasses


class DeterminePilot(object):
    '''
    Determine the pilot flying using available parameters within a section of
    flight (expects takeoff or landing). The order is:

    1. Pilot Flying parameter (Airbus AFPS analysis of sidestick movement)
    2. Pitch / Roll inputs where signals are recorded for Capt/FO seperately
    3. Control Column force where one force is at least ?% greater than the
       other
    4. Autopilot usage attributed to the Capt or FO's side


    Note: Use of VHF is not used as convention is:
    * VHF 1/L is the primary set for ATC comms
    * VHF 2/R is for company use, aircraft to aircraft and secondary ATC comms
    * VHF 3/C is for ACARS data comms / backup
    '''

    def _autopilot_engaged(self, ap1, ap2):
        '''
        This is a best guess assuming that if AP1 is in use, the Captain is PF.

        When the FO is PF, the right autopilot should be used. When the Capt
        is PF, the autopilot should alternate vetween Left and Center
        autopilots (to detect potential defects in center autopilot)

        TODO: Support Center auto-pilot usage
        '''
        if ap1 and (not ap2):
            return 'Captain'
        if (not ap1) and ap2:
            return 'First Officer'
        return None

    def _controls_changed(self, slice_, pitch, roll):
        # Check if either pitch or roll changed during provided slice:
        return pitch[slice_].ptp() > settings.CONTROLS_IN_USE_TOLERANCE or \
            roll[slice_].ptp() > settings.CONTROLS_IN_USE_TOLERANCE

    def _control_column_in_use(self, cc_capt, cc_fo, phase):
        '''
        Check if control column is used by Captain or FO.

        The forces are typically 2 or 3 lbf at takeoff, 6 or 7 at landing.
        Find the larger force of the two, checking they are at least a bit
        different as both move together on the B737-NG.
        '''
        force_ratio = cc_capt[phase.slice].ptp() / cc_fo[phase.slice].ptp()
        if force_ratio > settings.CONTROL_COLUMN_IN_USE_RATIO:
            self.info('Found Captain with force_ratio of %f', force_ratio)
            return 'Captain'
        elif (1.0/force_ratio) > settings.CONTROL_COLUMN_IN_USE_RATIO:
            self.info('Found First Officer with force_ratio of %f',
                      1.0/force_ratio)
            return 'First Officer'
        else:
            # No change in captain or first officer control columns:
            self.warning(
                "Neither captain's nor first officer's control column "
                "changes during '%s' slice.", phase.name)
            return None

    def _controls_in_use(self, pitch_capt, pitch_fo, roll_capt, roll_fo,
                         phase):
        capt_flying = self._controls_changed(phase.slice, pitch_capt,
                                             roll_capt)
        fo_flying = self._controls_changed(phase.slice, pitch_fo, roll_fo)

        # 1. Cannot determine who is flying - both sets of controls have input:
        if capt_flying and fo_flying:
            self.warning(
                "Cannot determine whether captain or first officer "
                "was at the controls because both controls change during '%s' "
                "slice.", phase.name)
            return None

        # 2. The captain was flying the aircraft:
        if capt_flying:
            return 'Captain'

        # 3. The first officer was flying the aircraft:
        if fo_flying:
            return 'First Officer'

        # 4. No change in captain or first officer controls:
        self.warning(
            "Both captain and first officer controls do not change "
            "during '%s' slice.", phase.name)
        return None

    def _determine_pilot(self, pilot_flying, pitch_capt, pitch_fo, roll_capt,
                         roll_fo, cc_capt, cc_fo, phase, ap1, ap2):

        if pilot_flying:
            # this is the most reliable measurement, use this and no other
            pf = pilot_flying.array[phase.slice]
            pf[pf == '-'] = np.ma.masked
            return most_common_value(pf)

        #FIXME: Skip over the Pitch and Control Column parts!
        # 1. Check for change in pitch and roll controls during the phase:
        if all((pitch_capt, pitch_fo, roll_capt, roll_fo, phase)):
            pilot = self._controls_in_use(
                pitch_capt.array, pitch_fo.array, roll_capt.array,
                roll_fo.array, phase)
            if pilot:
                return pilot

        # 1. Check for changes in control column during the phase:
        if all((cc_capt, cc_fo, phase)):
            pilot = self._control_column_in_use(cc_capt.array, cc_fo.array,
                                                phase)
            if pilot:
                return pilot

        # 2. Check which autopilot is engaged:
        if all((ap1, ap2)):
            pilot = self._autopilot_engaged(ap1, ap2)
            if pilot:
                return pilot

        return None


##############################################################################


class InvalidFlightType(Exception):
    def __init__(self, flight_type):
        self.flight_type = flight_type
        super(InvalidFlightType, self).__init__(flight_type)


class AnalysisDatetime(FlightAttributeNode):
    "Datetime flight was analysed (local datetime)"
    name = 'FDR Analysis Datetime'

    def derive(self, start_datetime=A('Start Datetime')):
        '''
        Every derive method requires at least one dependency. Since this class
        should always derive a flight attribute, 'Start Datetime' is its only
        dependency as it will always be present, though it is unused.
        '''
        self.set_flight_attr(datetime.utcnow().replace(tzinfo=pytz.utc))


class DestinationAirport(FlightAttributeNode):
    "Datetime flight was analysed (local datetime)"
    name = 'FDR Destination Airport'

    @classmethod
    def can_operate(cls, available):
        return any_of(['Destination', 'AFR Destination Airport'], available)

    def derive(self, dest=P('Destination'),
               afr_dest=A('AFR Destination Airport')):
        '''
        Requires an ASCII destination parameter recording either the airport's
        ICAO or IATA code.
        '''
        if not dest or dest.array.dtype.type is not np.string_:
            if afr_dest:
                self.set_flight_attr(afr_dest.value)
            return

        # XXX: Slow, but Destination should be sampled infrequently.
        value, count = next(reversed(sorted(Counter(dest.array).items(),
                                            key=itemgetter(1))))
        # Q: Should we validate that the value is 3-4 characters long?
        if value is np.ma.masked or count < len(dest.array) * 0.45:
            return

        api = get_api_handler(settings.API_HANDLER)
        try:
            airport = api.get_airport(value)
        except NotFoundError:
            self.warning('No destination airport found for %s.', value)
            return

        self.debug('Detected destination airport: %s', airport)
        self.set_flight_attr(airport)


class Duration(FlightAttributeNode):
    "Duration of the flight (between takeoff and landing) in seconds"
    name = 'FDR Duration'

    def derive(self, takeoff_dt=A('FDR Takeoff Datetime'),
               landing_dt=A('FDR Landing Datetime')):
        if landing_dt.value and takeoff_dt.value:
            duration = landing_dt.value - takeoff_dt.value
            self.set_flight_attr(duration.total_seconds())  # py2.7
        else:
            self.set_flight_attr(None)
            return


class FlightID(FlightAttributeNode):
    "Flight ID if provided via a known input attribute"
    name = 'FDR Flight ID'

    def derive(self, flight_id=A('AFR Flight ID')):
        self.set_flight_attr(flight_id.value)


class FlightNumber(FlightAttributeNode):
    """
    Returns String representation of the integer Flight Number value.

    Raises ValueError if negative value in array or too great a variance in
    array values.
    """
    "Airline route flight number"
    name = 'FDR Flight Number'

    def derive(self, num=P('Flight Number')):
        # Q: Should we validate the flight number?
        if num.array.dtype.type is np.string_:
            # XXX: Slow, but Flight Number should be sampled infrequently.
            value, count = next(reversed(sorted(Counter(num.array).items(),
                                                key=itemgetter(1))))
            if value is not np.ma.masked and count > len(num.array) * 0.45:
                self.set_flight_attr(value)
            return
        # Values of 0 are invalid flight numbers
        array = np.ma.masked_less_equal(num.array, 0)
        # Ignore masked values
        compressed_array = array.compressed()
        _, minvalue = min_value(compressed_array)
        if minvalue < 0:
            self.warning(
                "'%s' only supports unsigned (positive) values > 0, "
                "but none were found. Cannot determine flight number",
                self.name)
            self.set_flight_attr(None)
            return

        # note reverse of value, index from max_value due to bincount usage.
        value, count = max_value(
            np.bincount(compressed_array.astype(np.integer)))
        if count > len(compressed_array) * 0.45:
            # this value accounts for at least 45% of the values in the array
            self.set_flight_attr(str(int(value)))
        else:
            self.warning("Only %d out of %d flight numbers were the same."
                         " Flight Number attribute will be set as None.",
                         count or 0, len(num.array))
            self.set_flight_attr(None)
            return


class LandingAirport(FlightAttributeNode):
    '''
    The airport that the aircraft landed at determined from the flight data if
    possible, otherwise falling back to information provided in the achieved
    flight record.
    '''

    name = 'FDR Landing Airport'

    @classmethod
    def can_operate(cls, available):
        '''
        We can determine a landing airport in one of two ways:

        1. Find the nearest airport to the coordinates at landing.
        2. Use the airport data provided in the achieved flight record.
        '''
        return 'AFR Landing Airport' in available or all((
            'Latitude At Touchdown' in available,
            'Longitude At Touchdown' in available,
        ))

    def derive(self,
               land_lat=KPV('Latitude At Touchdown'),
               land_lon=KPV('Longitude At Touchdown'),
               land_afr_apt=A('AFR Landing Airport')):
        '''
        '''
        # 1. If we have latitude and longitude, look for the nearest airport:
        if land_lat and land_lon:
            lat = land_lat.get_last()
            lon = land_lon.get_last()
            if lat and lon:
                api = get_api_handler(settings.API_HANDLER)
                try:
                    airport = api.get_nearest_airport(lat.value, lon.value)
                except NotFoundError:
                    msg = 'No landing airport found near coordinates (%f, %f).'
                    self.warning(msg, lat.value, lon.value)
                    # No airport was found, so fall through and try AFR.
                else:
                    self.debug('Detected landing airport: %s', airport)
                    self.set_flight_attr(airport)
                    return  # We found an airport, so finish here.
            else:
                self.warning('No coordinates for looking up landing airport.')
                # No suitable coordinates, so fall through and try AFR.

        # 2. If we have an airport provided in achieved flight record, use it:
        if land_afr_apt:
            airport = land_afr_apt.value
            self.debug('Using landing airport from AFR: %s', airport)
            self.set_flight_attr(airport)
            return  # We found an airport in the AFR, so finish here.

        # 3. After all that, we still couldn't determine an airport...
        self.error('Unable to determine airport at landing!')
        self.set_flight_attr(None)


class LandingRunway(FlightAttributeNode):
    '''
    The runway that the aircraft landed at determined from the flight data if
    possible, otherwise falling back to information provided in the achieved
    flight record.
    '''

    name = 'FDR Landing Runway'

    @classmethod
    def can_operate(cls, available):
        '''
        We can determine a landing runway in a number of ways:

        1. Imprecisely using airport and heading during landing.
        2. Precisely using airport, heading and coordinates at landing.
        3. Use the runway data provided in the achieved flight record.
        '''
        minimum = all((
            'FDR Landing Airport' in available,
            'Heading During Landing' in available,
            'Approach And Landing' in available,
        ))

        fallback = 'AFR Landing Runway' in available

        return minimum or fallback

    def derive(self,
               land_fdr_apt=A('FDR Landing Airport'),
               land_afr_rwy=A('AFR Landing Runway'),
               land_hdg=KPV('Heading During Landing'),
               land_lat=KPV('Latitude At Touchdown'),
               land_lon=KPV('Longitude At Touchdown'),
               precision=A('Precise Positioning'),
               approaches=S('Approach And Landing'),
               ils_freq_on_app=KPV('ILS Frequency During Approach')):
        '''
        '''
        fallback = False
        precise = bool(getattr(precision, 'value', False))

        try:
            airport = int(land_fdr_apt.value['id'])
        except (AttributeError, KeyError, TypeError, ValueError):
            self.warning('Invalid airport... Fallback to AFR Landing Runway.')
            fallback = True

        try:
            heading = land_hdg.get_last().value
            if heading is None:
                raise ValueError
        except (AttributeError, ValueError):
            self.warning('Invalid heading... Fallback to AFR Landing Runway.')
            fallback = True

        try:
            landing = approaches.get_last()
            if landing is None:
                raise ValueError
        except (AttributeError, ValueError):
            self.warning('No approaches... Fallback to AFR Landing Runway.')
            # Don't set fallback - can still attempt to use heading only...

        # 1. If we have airport and heading, look for the nearest runway:
        if not fallback:
            kwargs = {}

            # The last approach is assumed to be the landing.
            # XXX: Last approach may not be landing for partial data?!
            if ils_freq_on_app:
                if landing.start_edge:
                    ils_app_slice = slice(landing.start_edge, landing.slice.stop)
                else:
                    ils_app_slice = landing.slice
                ils_freq = ils_freq_on_app.get_last(within_slice=ils_app_slice)
                if ils_freq:
                    kwargs.update(ils_freq=ils_freq.value)

            # We only provide coordinates when looking up a landing runway if
            # the recording of latitude and longitude on the aircraft is
            # precise. Inertial recordings are too inaccurate to pinpoint the
            # correct runway and we use ILS frequencies if possible to get a
            # more exact match.
            if precise and landing and land_lat and land_lon:
                lat = land_lat.get_last(within_slice=landing.slice)
                lon = land_lon.get_last(within_slice=landing.slice)
                if lat and lon:
                    kwargs.update(
                        latitude=lat.value,
                        longitude=lon.value,
                    )
                else:
                    self.warning('No coordinates for landing runway lookup.')
            else:
                kwargs.update(hint='landing')

            api = get_api_handler(settings.API_HANDLER)
            try:
                runway = api.get_nearest_runway(airport, heading, **kwargs)
            except NotFoundError:
                msg = 'No runway found for airport #%d @ %03.1f deg with %s.'
                self.warning(msg, airport, heading, kwargs)
                # No runway was found, so fall through and try AFR.
                if 'ils_freq' in kwargs:
                    # This is a trap for airports where the ILS data is not
                    # available, but the aircraft approached with the ILS
                    # tuned. A good prompt for an omission in the database.
                    self.warning('Fix database? No runway but ILS was tuned.')
            else:
                self.debug('Detected landing runway: %s', runway)
                self.set_flight_attr(runway)
                return  # We found a runway, so finish here.

        # 2. If we have a runway provided in achieved flight record, use it:
        if land_afr_rwy:
            runway = land_afr_rwy.value
            self.debug('Using landing runway from AFR: %s', runway)
            self.set_flight_attr(runway)
            return  # We found a runway in the AFR, so finish here.

        # 3. After all that, we still couldn't determine a runway...
        self.error('Unable to determine runway at landing!')
        self.set_flight_attr(None)


class OffBlocksDatetime(FlightAttributeNode):
    "Datetime when moving away from Gate/Blocks"
    name = 'FDR Off Blocks Datetime'

    def derive(self, turning=S('Turning On Ground'),
               start_datetime=A('Start Datetime')):
        first_turning = turning.get_first()
        if first_turning:
            off_blocks_datetime = datetime_of_index(start_datetime.value,
                                                    first_turning.slice.start,
                                                    turning.hz)
            self.set_flight_attr(off_blocks_datetime)
        else:
            self.set_flight_attr(None)


class OnBlocksDatetime(FlightAttributeNode):
    "Datetime when moving away from Gate/Blocks"
    name = 'FDR On Blocks Datetime'

    def derive(self, turning=S('Turning On Ground'),
               start_datetime=A('Start Datetime')):
        last_turning = turning.get_last()
        if last_turning:
            on_blocks_datetime = datetime_of_index(start_datetime.value,
                                                   last_turning.slice.stop,
                                                   turning.hz)
            self.set_flight_attr(on_blocks_datetime)
        else:
            self.set_flight_attr(None)


class TakeoffAirport(FlightAttributeNode):
    '''
    The airport that the aircraft took off from determined from the flight data
    if possible, otherwise falling back to information provided in the achieved
    flight record.
    '''

    name = 'FDR Takeoff Airport'

    @classmethod
    def can_operate(cls, available):
        '''
        We can determine a takeoff airport in one of three ways:

        1. Find the nearest airport to the coordinates at takeoff.
        2. Use the airport data provided in the achieved flight record.
        3. If segmetn does not takeoff eg RTO use coordinates off blocks
        '''
        complete_flight = all((
            'Latitude At Liftoff' in available,
            'Longitude At Liftoff' in available,
        ))
        afr = 'AFR Takeoff Airport' in available
        other_segments = all((
            'Latitude Off Blocks' in available,
            'Longitude Off Blocks' in available,
        ))
        return complete_flight or afr or other_segments

    def lookup_airport(self, lat_source, lon_source):
        lat = lat_source.get_first()
        lon = lon_source.get_first()
        if lat and lon:
            api = get_api_handler(settings.API_HANDLER)
            try:
                airport = api.get_nearest_airport(lat.value, lon.value)
            except NotFoundError:
                msg = 'No takeoff airport found near coordinates (%f, %f).'
                self.warning(msg, lat.value, lon.value)
                # No airport was found, so fall through and try AFR.
            else:
                self.debug('Detected takeoff airport: %s', airport)
                self.set_flight_attr(airport)
                return True  # We found an airport, so finish here.
        else:
            self.warning('No coordinates for looking up takeoff airport.')
            # No suitable coordinates, so fall through and try AFR.

    def derive(self,
               toff_lat=KPV('Latitude At Liftoff'),
               toff_lon=KPV('Longitude At Liftoff'),
               toff_afr_apt=A('AFR Takeoff Airport'),
               off_block_lat=KPV('Latitude Off Blocks'),
               off_block_lon=KPV('Longitude Off Blocks'),):
        '''
        '''
        # 1. If we have latitude and longitude, look for the nearest airport:
        if toff_lat and toff_lon:
            success = self.lookup_airport(toff_lat, toff_lon)
            if success:
                return

        # 2. If we have an airport provided in achieved flight record, use it:
        if toff_afr_apt:
            airport = toff_afr_apt.value
            self.debug('Using takeoff airport from AFR: %s', airport)
            self.set_flight_attr(airport)
            return  # We found an airport in the AFR, so finish here.

        # 3. If we have coordinates of Aircraft moving off Blocks look for the nearest airport:
        if off_block_lat and off_block_lon:
            success = self.lookup_airport(off_block_lat, off_block_lon)
            if success:
                return

        # 4. After all that, we still couldn't determine an airport...
        self.error('Unable to determine airport at takeoff!')
        self.set_flight_attr(None)


class TakeoffDatetime(FlightAttributeNode):
    '''
    Datetime at takeoff (first liftoff) or as close to this as possible.

    If no takeoff the following points in time will be used:
        Rejected Takeoff - Start of Rejected Takeoff phase
        Ground Run - Off Blocks
        No Movement - start of data
    '''
    name = 'FDR Takeoff Datetime'

    @classmethod
    def can_operate(cls, available):
        return 'Start Datetime' in available

    def derive(self, liftoff=KTI('Liftoff'), rto=S('Rejected Takeoff'),
               off_blocks=KTI('Off Blocks'), start_dt=A('Start Datetime')):

        if liftoff:
            # Flight - use first liftoff index
            first_liftoff = liftoff.get_first()
            liftoff_index = first_liftoff.index
            frequency = liftoff.frequency
        elif rto:
            # RTO - use start index of first RTO
            first_rto = rto.get_first()
            liftoff_index = first_rto.slice.start
            frequency = rto.frequency
        elif off_blocks:
            # Ground Only - use first off blocks index
            first_off_blocks = off_blocks.get_first()
            liftoff_index = first_off_blocks.index
            frequency = off_blocks.frequency
        else:
            # Incomplete - use start of data
            liftoff_index = 0
            frequency = 1

        takeoff_dt = datetime_of_index(start_dt.value, liftoff_index,
                                       frequency=frequency)
        self.set_flight_attr(takeoff_dt)


class TakeoffFuel(FlightAttributeNode):
    "Weight of Fuel in KG at point of Takeoff"
    name = 'FDR Takeoff Fuel'

    @classmethod
    def can_operate(cls, available):
        return 'AFR Takeoff Fuel' in available or \
               'Fuel Qty At Liftoff' in available

    def derive(self, afr_takeoff_fuel=A('AFR Takeoff Fuel'),
               liftoff_fuel_qty=KPV('Fuel Qty At Liftoff')):
        if afr_takeoff_fuel:
            #TODO: Validate that the AFR record is more accurate than the
            #flight data if available.
            self.set_flight_attr(afr_takeoff_fuel.value)
        else:
            fuel_qty_kpv = liftoff_fuel_qty.get_first()
            if fuel_qty_kpv:
                self.set_flight_attr(fuel_qty_kpv.value)


class TakeoffGrossWeight(FlightAttributeNode):
    "Aircraft Gross Weight in KG at point of Takeoff"
    name = 'FDR Takeoff Gross Weight'

    def derive(self, liftoff_gross_weight=KPV('Gross Weight At Liftoff')):
        first_gross_weight = liftoff_gross_weight.get_first()
        if first_gross_weight:
            self.set_flight_attr(first_gross_weight.value)
        else:
            # There is not a 'Gross Weight At Liftoff' KPV. Since it is sourced
            # from 'Gross Weight Smoothed', gross weight at liftoff should not
            # be masked.
            self.warning("No '%s' KPVs, '%s' attribute will be None.",
                         liftoff_gross_weight.name, self.name)
            self.set_flight_attr(None)


# FIXME: Check parameters for pitch and roll for captain and first officer!
#        What about 'Pitch Command (*)' and 'Sidestick [Pitch|Roll] (*)'?
# FIXME: This code does not identify the pilot correctly. Roll (FO) is the roll
#        attitude from the right side instrument, not the Airbus first officer
#        sidestick roll input. Needs a rewrite.
class TakeoffPilot(FlightAttributeNode, DeterminePilot):
    '''
    Pilot flying at takeoff - may be the captain, first officer or none.
    '''

    name = 'FDR Takeoff Pilot'

    @classmethod
    def can_operate(cls, available):
        pilot_flying = all_of((
            'Pilot Flying',
            'Takeoff',
        ), available)
        controls = all_of((
            'Pitch (Capt)',
            'Pitch (FO)',
            'Roll (Capt)',
            'Roll (FO)',
            'Takeoff',
        ), available)
        forces = all_of((
            'Control Column Force (Capt)',
            'Control Column Force (FO)',
            'Takeoff',
        ), available)
        autopilot = all_of((
            'AP (1) Engaged',
            'AP (2) Engaged',
            'Liftoff',
            # Optional: 'AP (3) Engaged'
        ), available)
        return pilot_flying or controls or forces or autopilot

    def derive(self,
               pilot_flying=M('Pilot Flying'),
               pitch_capt=P('Pitch (Capt)'),
               pitch_fo=P('Pitch (FO)'),
               roll_capt=P('Roll (Capt)'),
               roll_fo=P('Roll (FO)'),
               cc_capt=P('Control Column Force (Capt)'),
               cc_fo=P('Control Column Force (FO)'),
               ap1_eng=M('AP (1) Engaged'),
               ap2_eng=M('AP (2) Engaged'),
               takeoffs=S('Takeoff'),
               liftoffs=KTI('Liftoff'),
               rejected_toffs=S('Rejected Takeoff')):

        #TODO: Tidy
        phase = takeoffs or rejected_toffs or None
        if phase is None:
            # Nothing to do as no attempt to takeoff
            return
        lift = liftoffs.get_first() if liftoffs else None
        if lift and ap1_eng and ap2_eng:
            # check AP state at the floored index (just before lift)
            ap1 = ap1_eng.array[lift.index] == 'Engaged'
            ap2 = ap2_eng.array[lift.index] == 'Engaged'
        else:
            ap1 = ap2 = None
        args = (pilot_flying, pitch_capt, pitch_fo, roll_capt, roll_fo,
                cc_capt, cc_fo, phase.get_first(), ap1, ap2)
        self.set_flight_attr(self._determine_pilot(*args))


class TakeoffRunway(FlightAttributeNode):
    '''
    The runway that the aircraft took off from determined from the flight data
    if possible, otherwise falling back to information provided in the achieved
    flight record.
    '''

    name = 'FDR Takeoff Runway'

    @classmethod
    def can_operate(cls, available):
        '''
        We can determine a takeoff runway in a number of ways:

        1. Imprecisely using airport and heading during takeoff.
        2. Precisely using airport, heading and coordinates at takeoff.
        3. Use the runway data provided in the achieved flight record.
        '''
        minimum = all((
            'FDR Takeoff Airport' in available,
            'Heading During Takeoff' in available,
        ))

        fallback = 'AFR Takeoff Runway' in available

        return minimum or fallback

    def derive(self,
               toff_fdr_apt=A('FDR Takeoff Airport'),
               toff_afr_rwy=A('AFR Takeoff Runway'),
               toff_hdg=KPV('Heading During Takeoff'),
               toff_lat=KPV('Latitude At Liftoff'),
               toff_lon=KPV('Longitude At Liftoff'),
               precision=A('Precise Positioning')):
        '''
        '''
        fallback = False
        precise = bool(getattr(precision, 'value', False))

        try:
            airport = int(toff_fdr_apt.value['id'])
        except (AttributeError, KeyError, TypeError, ValueError):
            self.warning('Invalid airport... Fallback to AFR Takeoff Runway.')
            fallback = True

        try:
            heading = toff_hdg.get_first().value
            if heading is None:
                raise ValueError
        except (AttributeError, ValueError):
            self.warning('Invalid heading... Fallback to AFR Takeoff Runway.')
            fallback = True

        # 1. If we have airport and heading, look for the nearest runway:
        if not fallback:
            kwargs = {}

            # Even if we do not have precise latitude and longitude
            # information, we still use this for the takeoff runway detection
            # as it is often accurate at the start of a flight, and in the
            # absence of an ILS tuned frequency we have no better option. (We
            # did consider using the last direction of turn onto the runway,
            # but this would require an airport database with terminal and
            # taxiway details that was not felt justified).
            if toff_lat and toff_lon:
                lat = toff_lat.get_first()
                lon = toff_lon.get_first()
                if lat and lon:
                    kwargs.update(
                        latitude=lat.value,
                        longitude=lon.value,
                    )
                else:
                    self.warning('No coordinates for takeoff runway lookup.')
            if not precise:
                kwargs.update(hint='takeoff')

            api = get_api_handler(settings.API_HANDLER)
            try:
                runway = api.get_nearest_runway(airport, heading, **kwargs)
            except NotFoundError:
                msg = 'No runway found for airport #%d @ %03.1f deg with %s.'
                self.warning(msg, airport, heading, kwargs)
                # No runway was found, so fall through and try AFR.
            else:
                self.debug('Detected takeoff runway: %s', runway)
                self.set_flight_attr(runway)
                return  # We found a runway, so finish here.

        # 2. If we have a runway provided in achieved flight record, use it:
        if toff_afr_rwy:
            runway = toff_afr_rwy.value
            self.debug('Using takeoff runway from AFR: %s', runway)
            self.set_flight_attr(runway)
            return  # We found a runway in the AFR, so finish here.

        # 3. After all that, we still couldn't determine a runway...
        self.error('Unable to determine runway at takeoff!')
        self.set_flight_attr(None)


class FlightType(FlightAttributeNode):
    '''
    Type of flight flown
    '''
    name = 'FDR Flight Type'

    class Type(object):
        '''
        Type of flight.
        '''
        COMMERCIAL = 'COMMERCIAL'
        COMPLETE = 'COMPLETE'
        INCOMPLETE = 'INCOMPLETE'
        ENGINE_RUN_UP = 'ENGINE_RUN_UP'
        GROUND_RUN = 'GROUND_RUN'
        REJECTED_TAKEOFF = 'REJECTED_TAKEOFF'
        TEST = 'TEST'
        TRAINING = 'TRAINING'
        FERRY = 'FERRY'
        POSITIONING = 'POSITIONING'
        LINE_TRAINING = 'LINE_TRAINING'

    @classmethod
    def can_operate(cls, available):
        return any_of(cls.get_dependency_names(), available)

    def derive(self, afr_type=A('AFR Type'), fast=S('Fast'), mobile=S('Mobile'),
               liftoffs=KTI('Liftoff'), touchdowns=KTI('Touchdown'),
               touch_and_gos=S('Touch And Go'), rejected_to=S('Rejected Takeoff'),
               eng_start=KTI('Eng Start')):
        '''
        TODO: Detect MID_FLIGHT.
        '''
        afr_type = afr_type.value if afr_type else None

        if liftoffs and not touchdowns:
            # In the air without having touched down.
            self.warning("'Liftoff' KTI exists without 'Touchdown'.")
            raise InvalidFlightType('LIFTOFF_ONLY')
            #self.set_flight_attr('LIFTOFF_ONLY')
            #return
        elif not liftoffs and touchdowns:
            # In the air without having lifted off.
            self.warning("'Touchdown' KTI exists without 'Liftoff'.")
            raise InvalidFlightType('TOUCHDOWN_ONLY')
            #self.set_flight_attr('TOUCHDOWN_ONLY')
            #return

        if liftoffs and touchdowns:
            first_touchdown = touchdowns.get_first()
            first_liftoff = liftoffs.get_first()
            if first_touchdown.index < first_liftoff.index:
                # Touchdown before having lifted off, data must be INCOMPLETE.
                self.warning("'Touchdown' KTI index before 'Liftoff'.")
                raise InvalidFlightType('TOUCHDOWN_BEFORE_LIFTOFF')
                #self.set_flight_attr('TOUCHDOWN_BEFORE_LIFTOFF')
                #return
            last_touchdown = touchdowns.get_last()  # TODO: Delete line.
            if touch_and_gos:
                last_touchdown = touchdowns.get_last()
                last_touch_and_go = touch_and_gos.get_last()
                if last_touchdown.index <= last_touch_and_go.index:
                    self.warning("A 'Touch And Go' KTI exists after the last "
                                 "'Touchdown'.")
                    raise InvalidFlightType('LIFTOFF_ONLY')
                    #self.set_flight_attr('LIFTOFF_ONLY')
                    #return

            if afr_type in [FlightType.Type.FERRY,
                            FlightType.Type.LINE_TRAINING,
                            FlightType.Type.POSITIONING,
                            FlightType.Type.TEST,
                            FlightType.Type.TRAINING]:
                flight_type = afr_type
            else:
                flight_type = FlightType.Type.COMPLETE
        elif rejected_to:
            # Rejected takeoff but no takeoff or landing
            flight_type = FlightType.Type.REJECTED_TAKEOFF
        elif fast:
            # Midflight as no takeoff, rejected takeoff or landing but went fast
            flight_type = FlightType.Type.INCOMPLETE
        elif mobile:
            # The aircraft moved on the ground.
            flight_type = FlightType.Type.GROUND_RUN
        elif eng_start:
            # Engines were running at some point
            flight_type = FlightType.Type.ENGINE_RUN_UP
        else:
            # TODO: not detected flight type should we fall back to No Movement?
            # should we raise an error
            flight_type = FlightType.Type.INCOMPLETE
        self.set_flight_attr(flight_type)

#Q: Not sure if we can identify Destination from the data?
##class DestinationAirport(FlightAttributeNode):
    ##""
    ##def derive(self):
        ##return NotImplemented
                    ##{'id':9456, 'name':'City. Airport'}


class LandingDatetime(FlightAttributeNode):
    """ Datetime at landing (final touchdown) or as close to this as possible.
    If no landing (incomplete flight / ground run) store None.
    """
    name = 'FDR Landing Datetime'

    def derive(self, start_datetime=A('Start Datetime'),
               touchdown=KTI('Touchdown')):
        last_touchdown = touchdown.get_last()
        if not last_touchdown:
            self.set_flight_attr(None)
            return
        landing_datetime = datetime_of_index(start_datetime.value,
                                             last_touchdown.index,
                                             frequency=touchdown.frequency)
        self.set_flight_attr(landing_datetime)


class LandingFuel(FlightAttributeNode):
    "Weight of Fuel in KG at point of Touchdown"
    name = 'FDR Landing Fuel'

    @classmethod
    def can_operate(cls, available):
        return 'AFR Landing Fuel' in available or \
               'Fuel Qty At Touchdown' in available

    def derive(self, afr_landing_fuel=A('AFR Landing Fuel'),
               touchdown_fuel_qty=KPV('Fuel Qty At Touchdown')):
        if afr_landing_fuel:
            self.set_flight_attr(afr_landing_fuel.value)
        else:
            fuel_qty_kpv = touchdown_fuel_qty.get_last()
            if fuel_qty_kpv:
                self.set_flight_attr(fuel_qty_kpv.value)


class LandingGrossWeight(FlightAttributeNode):
    "Aircraft Gross Weight in KG at point of Landing"
    name = 'FDR Landing Gross Weight'

    def derive(self, touchdown_gross_weight=KPV('Gross Weight At Touchdown')):
        last_gross_weight = touchdown_gross_weight.get_last()
        if last_gross_weight:
            self.set_flight_attr(last_gross_weight.value)
        else:
            # There is not a 'Gross Weight At Touchdown' KPV. Since it is
            # sourced from 'Gross Weight Smoothed', gross weight at touchdown
            # should not be masked. Are there no Touchdown KTIs?
            self.warning("No '%s' KPVs, '%s' attribute will be None.",
                         touchdown_gross_weight.name, self.name)
            self.set_flight_attr(None)


# FIXME: Check parameters for pitch and roll for captain and first officer!
#        What about 'Pitch Command (*)' and 'Sidestick [Pitch|Roll] (*)'?
# FIXME: This code does not identify the pilot correctly. Roll (FO) is the roll
#        attitude from the right side instrument, not the Airbus first officer
#        sidestick roll input. Needs a rewrite.
class LandingPilot(FlightAttributeNode, DeterminePilot):
    '''
    Pilot flying at landing - may be the captain, first officer or none.
    '''

    name = 'FDR Landing Pilot'

    @classmethod
    def can_operate(cls, available):
        pilot_flying = all_of((
            'Pilot Flying',
            'Landing',
        ), available)
        controls = all_of((
            'Pitch (Capt)',
            'Pitch (FO)',
            'Roll (Capt)',
            'Roll (FO)',
            'Landing',
        ), available)
        forces = all_of((
            'Control Column Force (Capt)',
            'Control Column Force (FO)',
            'Landing',
        ), available)
        autopilot = all_of((
            'AP (1) Engaged',
            'AP (2) Engaged',
            'Touchdown',
            # Optional: 'AP (3) Engaged'
        ), available)
        return pilot_flying or controls or forces or autopilot

    def derive(self,
               pilot_flying=M('Pilot Flying'),
               pitch_capt=P('Pitch (Capt)'),
               pitch_fo=P('Pitch (FO)'),
               roll_capt=P('Roll (Capt)'),
               roll_fo=P('Roll (FO)'),
               cc_capt=P('Control Column Force (Capt)'),
               cc_fo=P('Control Column Force (FO)'),
               ap1_eng=M('AP (1) Engaged'),
               ap2_eng=M('AP (2) Engaged'),
               landings=S('Landing'),
               touchdowns=KTI('Touchdown')):

        phase = landings.get_last() if landings else None
        tdwn = touchdowns.get_last() if touchdowns else None
        if tdwn and ap1_eng and ap2_eng:
            # check AP state at the floored index (just before tdwn)
            ap1 = ap1_eng.array[tdwn.index] == 'Engaged'
            ap2 = ap2_eng.array[tdwn.index] == 'Engaged'
        else:
            ap1 = ap2 = None
        args = (pilot_flying, pitch_capt, pitch_fo, roll_capt, roll_fo,
                cc_capt, cc_fo, phase, ap1, ap2)
        self.set_flight_attr(self._determine_pilot(*args))


class Version(FlightAttributeNode):
    "Version of code used for analysis"
    name = 'FDR Version'

    def derive(self, start_datetime=A('Start Datetime')):
        '''
        Every derive method requires at least one dependency. Since this class
        should always derive a flight attribute, 'Start Datetime' is its only
        dependency as it will always be present, though it is unused.
        '''
        self.set_flight_attr(__version__)


##############################################################################
# vim:et:ft=python:nowrap:sts=4:sw=4:ts=4
