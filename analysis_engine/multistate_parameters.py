# -*- coding: utf-8 -*-

import math
import logging

import numpy as np

from pprint import pformat

from flightdatautilities import aircrafttables as at, units as ut

from hdfaccess.parameter import MappedArray

from analysis_engine.node import (
    A, MultistateDerivedParameterNode,
    M,
    P,
    S,
)
from analysis_engine.library import (
    all_of,
    any_of,
    calculate_flap,
    calculate_slat,
    calculate_surface_angle,
    datetime_of_index,
    find_edges_on_state_change,
    including_transition,
    index_at_value,
    index_closest_value,
    is_day,
    merge_masks,
    merge_sources,
    merge_two_parameters,
    moving_average,
    nearest_neighbour_mask_repair,
    np_ma_masked_zeros_like,
    np_ma_zeros_like,
    offset_select,
    repair_mask,
    runs_of_ones,
    second_window,
    slices_from_to,
    slices_remove_small_gaps,
    slices_remove_small_slices,
    step_values,
    vstack_params_where_state,
)
from settings import (
    MIN_CORE_RUNNING,
    MIN_FAN_RUNNING,
    MIN_FUEL_FLOW_RUNNING,
)

logger = logging.getLogger(name=__name__)


class APEngaged(MultistateDerivedParameterNode):
    '''
    Determines if *any* of the "AP (*) Engaged" parameters are recording the
    state of Engaged.

    This is a discrete with only the Engaged state.
    '''

    name = 'AP Engaged'
    values_mapping = {0: '-', 1: 'Engaged'}

    @classmethod
    def can_operate(cls, available):
        return any_of(cls.get_dependency_names(), available)

    def derive(self,
               ap1=M('AP (1) Engaged'),
               ap2=M('AP (2) Engaged'),
               ap3=M('AP (3) Engaged')):
        stacked = vstack_params_where_state(
            (ap1, 'Engaged'),
            (ap2, 'Engaged'),
            (ap3, 'Engaged'),
        )
        self.array = stacked.any(axis=0)
        self.offset = offset_select('mean', [ap1, ap2, ap3])


class APChannelsEngaged(MultistateDerivedParameterNode):
    '''
    Assess the number of autopilot systems engaged.

    Airbus and Boeing = 1 autopilot at a time except when "Land" mode
    selected when 2 (Dual) or 3 (Triple) can be engaged. Airbus favours only
    2 APs, Boeing is happier with 3 though some older types may only have 2.
    '''
    name = 'AP Channels Engaged'
    values_mapping = {0: '-', 1: 'Single', 2: 'Dual', 3: 'Triple'}

    @classmethod
    def can_operate(cls, available):
        return len(available) >= 2

    def derive(self,
               ap1=M('AP (1) Engaged'),
               ap2=M('AP (2) Engaged'),
               ap3=M('AP (3) Engaged')):
        stacked = vstack_params_where_state(
            (ap1, 'Engaged'),
            (ap2, 'Engaged'),
            (ap3, 'Engaged'),
        )
        self.array = stacked.sum(axis=0)
        self.offset = offset_select('mean', [ap1, ap2, ap3])


class APLateralMode(MultistateDerivedParameterNode):
    '''
    '''
    name = 'AP Lateral Mode'
    # Values and states match X-Plane visualisation model documentation.
    values_mapping = {
        0: '-',
        2: 'RWY',
        4: 'RWY TRK',
        6: 'NAV',
        14: 'LOC CAPT',
        16: 'LOC',
        20: 'APP NAV',
        22: 'ROLL OUT',
        24: 'LAND',
        64: 'HDG',
    }

    @classmethod
    def can_operate(cls, available):
        return any_of(('Lateral Mode Selected',
                       'Runway Mode Active',
                       'NAV Mode Active',
                       'ILS Localizer Capture Active',
                       'ILS Localizer Track Active',
                       'Roll Go Around Mode Active',
                       'Land Track Active',
                       'Heading Mode Active'), available)

    def derive(self,
               lateral_mode_selected=M('Lateral Mode Selected'),
               runway_mode_active=M('Runway Mode Active'),
               nav_mode_active=M('NAV Mode Active'),
               ils_localizer_capture_active=M('ILS Localizer Capture Active'),
               ils_localizer_track_active=M('ILS Localizer Track Active'),
               roll_go_around_mode_active=M('Roll Go Around Mode Active'),
               land_track_active=M('Land Track Active'),
               heading_mode_active=M('Heading Mode Active')):
        parameter = next(p for p in (lateral_mode_selected,
                                     runway_mode_active,
                                     nav_mode_active,
                                     ils_localizer_capture_active,
                                     ils_localizer_track_active,
                                     roll_go_around_mode_active,
                                     land_track_active,
                                     heading_mode_active) if p)
        self.array = np_ma_zeros_like(parameter.array)

        if lateral_mode_selected:
            self.array[lateral_mode_selected.array == 'Runway Mode Active'] = 'RWY'
            self.array[lateral_mode_selected.array == 'NAV Mode Active'] = 'NAV'
            self.array[lateral_mode_selected.array == 'ILS Localizer Capture Active'] = 'LOC CAPT',
        if runway_mode_active:
            self.array[runway_mode_active.array == 'Activated'] = 'RWY'
        if nav_mode_active:
            self.array[nav_mode_active.array == 'Activated'] = 'NAV'
        if ils_localizer_capture_active:
            self.array[ils_localizer_capture_active.array == 'Activated'] = 'LOC CAPT'
        if ils_localizer_track_active:
            self.array[ils_localizer_track_active.array == 'Activated'] = 'LOC'
        if roll_go_around_mode_active:
            self.array[roll_go_around_mode_active.array == 'Activated'] = 'ROLL OUT'
        if land_track_active:
            self.array[land_track_active.array == 'Activated'] = 'LAND'
        if heading_mode_active:
            self.array[heading_mode_active.array == 'Activated'] = 'HDG'


class APVerticalMode(MultistateDerivedParameterNode):
    '''
    '''
    name = 'AP Vertical Mode'
    # Values and states match X-Plane visualisation model documentation.
    values_mapping = {
        0: '-',
        2: 'SRS',
        4: 'CLB',
        6: 'DES',
        8: 'ALT CSTR CAPT',
        10: 'ALT CSTR',
        14: 'GS CAPT',
        16: 'GS',
        18: 'FINAL',
        22: 'FLARE',
        24: 'LAND',
        26: 'DES',  # geo path, A/THR mode SPEED
        64: 'OP CLB',
        66: 'OP DES',
        68: 'ALT CAPT',
        70: 'ALT',
        72: 'ALT CRZ',
        76: 'V/S',
        86: 'EXPED CLB',
        88: 'EXPED DES',
    }

    @classmethod
    def can_operate(cls, available):
        return any_of(('AT Active',
                       'Climb Mode Active',
                       'Longitudinal Mode Selected',
                       'ILS Glideslope Capture Active',
                       'ILS Glideslope Active',
                       'Flare Mode',
                       'Open Climb Mode',
                       'Open Descent Mode',
                       'Altitude Capture Mode',
                       'Altitude Mode',
                       'Expedite Climb Mode',
                       'Expedite Descent Mode'), available)

    def derive(self,
               at_active=M('AT Active'),
               climb_mode_active=M('Climb Mode Active'),
               longitudinal_mode_selected=M('Longitudinal Mode Selected'),
               ils_glideslope_capture_active=M('ILS Glideslope Capture Active'),
               ils_glideslope_active=M('ILS Glideslope Active'),
               flare_mode=M('Flare Mode'),
               open_climb_mode=M('Open Climb Mode'),
               open_descent_mode=M('Open Descent Mode'),
               altitude_capture_mode=M('Altitude Capture Mode'),
               altitude_mode=M('Altitude Mode'),
               expedite_climb_mode=M('Expedite Climb Mode'),
               expedite_descent_mode=M('Expedite Descent Mode')):
        parameter = next(p for p in (climb_mode_active,
                                     longitudinal_mode_selected,
                                     ils_glideslope_capture_active,
                                     ils_glideslope_active,
                                     flare_mode,
                                     at_active,
                                     open_climb_mode,
                                     open_descent_mode,
                                     altitude_capture_mode,
                                     altitude_mode,
                                     expedite_climb_mode,
                                     expedite_descent_mode) if p)
        self.array = np_ma_zeros_like(parameter.array)

        if at_active:
            self.array[at_active.array == 'Activated'] = 'DES'
        if climb_mode_active:
            self.array[climb_mode_active.array == 'Activated'] = 'CLB'
        if longitudinal_mode_selected:
            self.array[longitudinal_mode_selected.array == 'Altitude'] = 'ALT CSTR'
            self.array[longitudinal_mode_selected.array == 'Final Descent Mode'] = 'FINAL'
            self.array[longitudinal_mode_selected.array == 'Flare Mode'] = 'FLARE'
            self.array[longitudinal_mode_selected.array == 'Land Track Active'] = 'LAND'
            self.array[longitudinal_mode_selected.array == 'Vertical Speed Engaged'] = 'V/S'
        if ils_glideslope_capture_active:
            self.array[ils_glideslope_capture_active.array == 'Activated'] = 'GS CAPT'
        if ils_glideslope_active:
            self.array[ils_glideslope_active.array == 'Activated'] = 'GS'
        if flare_mode:
            self.array[flare_mode.array == 'Engaged'] = 'FLARE'
        if open_climb_mode:
            self.array[open_climb_mode.array == 'Activated'] = 'OP CLB'
        if open_descent_mode:
            self.array[open_descent_mode.array == 'Activated'] = 'OP DES'
        if altitude_capture_mode:
            self.array[altitude_capture_mode.array == 'Activated'] = 'ALT CAPT'
        if altitude_mode:
            self.array[altitude_mode.array == 'Activated'] = 'ALT'
        if expedite_climb_mode:
            self.array[expedite_climb_mode.array == 'Activated'] = 'EXPED CLB'
        if expedite_descent_mode:
            self.array[expedite_descent_mode.array == 'Activated'] = 'EXPED DES'


class APUOn(MultistateDerivedParameterNode):
    '''
    Combine APU (1) On and APU (2) On parameters.
    '''

    name = 'APU On'

    values_mapping = {0: '-', 1: 'On'}

    @classmethod
    def can_operate(cls, available):
        return any_of(('APU (1) On', 'APU (2) On'), available)

    def derive(self, apu_1=M('APU (1) On'), apu_2=M('APU (2) On')):
        self.array = vstack_params_where_state(
            (apu_1, 'On'),
            (apu_2, 'On'),
        ).any(axis=0)


class APURunning(MultistateDerivedParameterNode):
    '''
    Simple measure of APU status, suitable for plotting if you want an on/off
    measure. Used for fuel usage measurements.
    '''

    name = 'APU Running'

    values_mapping = {0: '-', 1: 'Running'}

    @classmethod
    def can_operate(cls, available):
        return any_of(('APU N1',
                       'APU Generator AC Voltage',
                       'APU Bleed Valve Open'), available)

    def derive(self, apu_n1=P('APU N1'),
               apu_voltage=P('APU Generator AC Voltage'),
               apu_bleed_valve_open=M('APU Bleed Valve Open')):
        if apu_n1:
            self.array = np.ma.where(apu_n1.array > 50.0, 'Running', '-')
        elif apu_voltage:
            # XXX: APU Generator AC Voltage > 100 volts.
            self.array = np.ma.where(apu_voltage.array > 100.0, 'Running', '-')
        else:
            self.array = apu_bleed_valve_open.array == 'Open'


class Configuration(MultistateDerivedParameterNode):
    '''
    Parameter for aircraft that use configuration. Reflects the actual state
    of the aircraft. See "Flap Lever" or "Flap Lever (Synthetic)" which show
    the physical lever detents selectable by the crew.

    Multi-state with the following mapping::

        %s

    Some values are based on footnotes in various pieces of documentation:

    - 2(a) corresponds to CONF 1*
    - 3(b) corresponds to CONF 2*

    Note: Does not use the Flap Lever position. This parameter reflects the
    actual configuration state of the aircraft rather than the intended state
    represented by the selected lever position.

    Note: Values that do not map directly to a required state are masked
    ''' % pformat(at.constants.AVAILABLE_CONF_STATES)
    values_mapping = at.constants.AVAILABLE_CONF_STATES
    align_frequency = 2

    @classmethod
    def can_operate(cls, available, manufacturer=A('Manufacturer'),
                    model=A('Model'), series=A('Series'), family=A('Family')):

        if manufacturer and not manufacturer.value == 'Airbus':
            return False

        if family and family.value in ('A300', 'A310'):
            return False

        if not all_of(('Slat', 'Flap', 'Model', 'Series', 'Family'), available):
            return False

        try:
            at.get_conf_angles(model.value, series.value, family.value)
        except KeyError:
            cls.warning("No conf angles available for '%s', '%s', '%s'.",
                        model.value, series.value, family.value)
            return False

        return True

    def derive(self, slat=M('Slat'), flap=M('Flap'), flaperon=M('Flaperon'),
               model=A('Model'), series=A('Series'), family=A('Family')):

        angles = at.get_conf_angles(model.value, series.value, family.value)
        self.array = MappedArray(np_ma_masked_zeros_like(flap.array, dtype=np.short),
                                 values_mapping=self.values_mapping)
        for (state, (s, f, a)) in angles.iteritems():
            condition = (flap.array == f)
            if s is not None:
                condition &= (slat.array == s)
            if a is not None:
                condition &= (flaperon.array == a)
            self.array[condition] = state

        # Repair the mask to smooth out transitions:
        nearest_neighbour_mask_repair(self.array, copy=False,
                                      repair_gap_size=(30 * self.hz),
                                      direction='backward')


class Daylight(MultistateDerivedParameterNode):
    '''
    Calculate Day or Night based upon Civil Twilight.

    FAA Regulation FAR 1.1 defines night as: "Night means the time between
    the end of evening civil twilight and the beginning of morning civil
    twilight, as published in the American Air Almanac, converted to local
    time.

    EASA EU OPS 1 Annex 1 item (76) states: 'night' means the period between
    the end of evening civil twilight and the beginning of morning civil
    twilight or such other period between sunset and sunrise as may be
    prescribed by the appropriate authority, as defined by the Member State;

    CAA regulations confusingly define night as 30 minutes either side of
    sunset and sunrise, then include a civil twilight table in the AIP.

    With these references, it was decided to make civil twilight the default.
    '''
    align = True
    # 1/4 is the minimum allowable frequency due to minimum data boundary
    # of 4 seconds.
    align_frequency = 1 / 4.0
    align_offset = 0.0

    values_mapping = {
        0: 'Night',
        1: 'Day'
    }

    def derive(self,
               latitude=P('Latitude Smoothed'),
               longitude=P('Longitude Smoothed'),
               start_datetime=A('Start Datetime'),
               duration=A('HDF Duration')):
        # Set default to 'Day'
        array_len = duration.value * self.frequency
        self.array = np.ma.ones(array_len)
        for step in xrange(int(array_len)):
            curr_dt = datetime_of_index(start_datetime.value, step, 1)
            lat = latitude.array[step]
            lon = longitude.array[step]
            if lat and lon:
                if not is_day(curr_dt, lat, lon):
                    # Replace values with Night
                    self.array[step] = 0
                else:
                    continue  # leave array as 1
            else:
                # either is masked or recording 0.0 which is invalid too
                self.array[step] = np.ma.masked


class DualInput(MultistateDerivedParameterNode):
    '''
    Determines whether input by both of the pilots has occurred.

    This parameter uses the 'Pilot Flying' derived multi-state parameter to
    determine who is considered to be the pilot flying the aircraft and then
    inspects whether the other pilot has made any significant sustained input.

    For Airbus aircraft, this requires us to check the angle of the sidestick.

    Note that the AFPS defines DUAL_INPUT as 0.5 degree deflection for more
    than 3 seconds. However, the A330/A340 have poor resolution so this
    threshold was increased to 1.7 degrees. SmartCockpit Flight Controls
    Sidestick priority logic declares a 2.0 degree deflection will trigger
    "SIDE STICK PRIORITY" lights on the glareshield and the "DUAL INPUT" voice
    message is activated. Therefore, the threshold used here is 2.0 degrees for
    3 seconds so all maximum sidestick angle KPVs measured during Dual Input
    will have a minimum of 2.0 degrees.

    This is not strictly speaking a warning as we have no record that anything
    was triggered in the cockpit.

    Reference was made to the following documentation to assist with the
    development of this algorithm:

    - A320 Flight Profile Specification
    - A321 Flight Profile Specification
    '''
    values_mapping = {0: '-', 1: 'Dual'}

    def derive(self,
               pilot=M('Pilot Flying'),
               stick_capt=P('Sidestick Angle (Capt)'),
               stick_fo=P('Sidestick Angle (FO)')):

        array = np_ma_zeros_like(pilot.array)
        array[pilot.array == 'Captain'] = stick_fo.array[pilot.array == 'Captain']
        array[pilot.array == 'First Officer'] = stick_capt.array[pilot.array == 'First Officer']
        array = np.ma.array(array > 2.0, mask=array.mask, dtype=int)

        slices = runs_of_ones(array)
        slices = slices_remove_small_gaps(slices, 15, self.hz)
        slices = slices_remove_small_slices(slices, 3, self.hz)

        dual = np_ma_zeros_like(array, dtype=np.short)
        for sl in slices:
            dual[sl] = 1
        self.array = dual


class Eng_1_Fire(MultistateDerivedParameterNode):
    '''
    Combine on ground and in air fire warnings.
    '''

    name = 'Eng (1) Fire'
    values_mapping = {0: '-', 1: 'Fire'}

    def derive(self,
               fire_gnd=M('Eng (1) Fire On Ground'),
               fire_air=M('Eng (1) Fire In Air')):

        self.array = vstack_params_where_state(
            (fire_gnd, 'Fire'),
            (fire_air, 'Fire'),
        ).any(axis=0)


class Eng_2_Fire(MultistateDerivedParameterNode):
    '''
    Combine on ground and in air fire warnings.
    '''

    name = 'Eng (2) Fire'
    values_mapping = {0: '-', 1: 'Fire'}

    def derive(self,
               fire_gnd=M('Eng (2) Fire On Ground'),
               fire_air=M('Eng (2) Fire In Air')):

        self.array = vstack_params_where_state(
            (fire_gnd, 'Fire'),
            (fire_air, 'Fire'),
        ).any(axis=0)


class Eng_3_Fire(MultistateDerivedParameterNode):
    '''
    Combine on ground and in air fire warnings.
    '''

    name = 'Eng (3) Fire'
    values_mapping = {0: '-', 1: 'Fire'}

    def derive(self,
               fire_gnd=M('Eng (3) Fire On Ground'),
               fire_air=M('Eng (3) Fire In Air')):

        self.array = vstack_params_where_state(
            (fire_gnd, 'Fire'),
            (fire_air, 'Fire'),
        ).any(axis=0)


class Eng_4_Fire(MultistateDerivedParameterNode):
    '''
    Combine on ground and in air fire warnings.
    '''

    name = 'Eng (4) Fire'
    values_mapping = {0: '-', 1: 'Fire'}

    def derive(self,
               fire_gnd=M('Eng (4) Fire On Ground'),
               fire_air=M('Eng (4) Fire In Air')):

        self.array = vstack_params_where_state(
            (fire_gnd, 'Fire'),
            (fire_air, 'Fire'),
        ).any(axis=0)


class Eng_Fire(MultistateDerivedParameterNode):
    '''
    Merges all the engine fire signals into one.
    '''
    name = 'Eng (*) Fire'
    values_mapping = {0: '-', 1: 'Fire'}

    @classmethod
    def can_operate(cls, available):
        return any_of(cls.get_dependency_names(), available)

    def derive(self,
               eng1=M('Eng (1) Fire'),
               eng2=M('Eng (2) Fire'),
               eng3=M('Eng (3) Fire'),
               eng4=M('Eng (4) Fire'),
               eng1_1l=M('Eng (1) Fire (1L)'),
               eng1_1r=M('Eng (1) Fire (1R)'),
               eng1_2l=M('Eng (1) Fire (2L)'),
               eng1_2r=M('Eng (1) Fire (2R)'),
               ):

        self.array = vstack_params_where_state(
            (eng1, 'Fire'), (eng2, 'Fire'),
            (eng3, 'Fire'), (eng4, 'Fire'),
            (eng1_1l, 'Fire'), (eng1_1r, 'Fire'),
            (eng1_2l, 'Fire'), (eng1_2r, 'Fire'),
        ).any(axis=0)


class Eng_Oil_Press_Warning(MultistateDerivedParameterNode):
    '''
    Combine all oil pressure (low) warning indications.
    '''

    name = 'Eng (*) Oil Press Warning'
    values_mapping = {0: '-', 1: 'Warning'}

    @classmethod
    def can_operate(cls, available):
        return any_of(cls.get_dependency_names(), available)

    def derive(self,
               eng1=P('Eng (1) Oil Press Low'),
               eng2=P('Eng (2) Oil Press Low'),
               eng3=P('Eng (3) Oil Press Low'),
               eng4=P('Eng (4) Oil Press Low'),
               ):

        self.array = vstack_params_where_state(
            (eng1, 'Low Press'),
            (eng2, 'Low Press'),
            (eng3, 'Low Press'),
            (eng4, 'Low Press'),
        ).any(axis=0)


class EngRunning(object):
    '''
    Abstract class for inheriting by EngRunning derived parameters.
    '''
    engnum = 0  # Replace with '2' for Eng (2)
    values_mapping = {
        0: 'Not Running',
        1: 'Running',
    }

    @classmethod
    def can_operate(cls, available):
        return 'Eng (%d) N1' % cls.engnum in available or \
               'Eng (%d) N2' % cls.engnum in available or \
               'Eng (%d) Np' % cls.engnum in available or \
               'Eng (%d) Fuel Flow' % cls.engnum in available

    def determine_running(self, eng_n1, eng_n2, eng_np, fuel_flow):
        '''
        TODO: Include Fuel cut-off switch if recorded?
        TODO: Confirm that all engines were recording for the N2 Min / Fuel Flow
        Min parameters - theoretically there could be only three engines in the
        frame for a four engine aircraft. Use "Engine Count".        
        '''
        if eng_np:
            # If it's got propellors, this overrides core engine measurements.
            return eng_np.array > MIN_FAN_RUNNING
        elif eng_n2 or fuel_flow:
            # Ideally have N2 and Fuel Flow with both available,
            # otherwise use just one source
            n2_running = eng_n2.array > MIN_CORE_RUNNING if eng_n2 \
                else np.ones_like(fuel_flow.array, dtype=bool)
            fuel_flowing = fuel_flow.array > MIN_FUEL_FLOW_RUNNING if fuel_flow \
                else np.ones_like(eng_n2.array, dtype=bool)
            return n2_running & fuel_flowing
        else:
            # Fall back on N1
            return eng_n1.array > MIN_FAN_RUNNING


class Eng1Running(EngRunning, MultistateDerivedParameterNode):
    '''
    Discrete parameter describing when the engine is running.
    '''
    engnum = 1
    name = 'Eng (1) Running'

    def derive(self,
               eng_n1=P('Eng (1) N1'),
               eng_n2=P('Eng (1) N2'),
               eng_np=P('Eng (1) Np'),
               fuel_flow=P('Eng (1) Fuel Flow')):
        self.array = self.determine_running(eng_n1, eng_n2, eng_np, fuel_flow)


class Eng2Running(EngRunning, MultistateDerivedParameterNode):
    '''
    Discrete parameter describing when the engine is running.
    '''
    engnum = 2
    name = 'Eng (2) Running'

    def derive(self,
               eng_n1=P('Eng (2) N1'),
               eng_n2=P('Eng (2) N2'),
               eng_np=P('Eng (2) Np'),
               fuel_flow=P('Eng (2) Fuel Flow')):
        self.array = self.determine_running(eng_n1, eng_n2, eng_np, fuel_flow)


class Eng3Running(EngRunning, MultistateDerivedParameterNode):
    '''
    Discrete parameter describing when the engine is running.
    '''
    engnum = 3
    name = 'Eng (3) Running'

    def derive(self,
               eng_n1=P('Eng (3) N1'),
               eng_n2=P('Eng (3) N2'),
               eng_np=P('Eng (3) Np'),
               fuel_flow=P('Eng (3) Fuel Flow')):
        self.array = self.determine_running(eng_n1, eng_n2, eng_np, fuel_flow)


class Eng4Running(EngRunning, MultistateDerivedParameterNode):
    '''
    Discrete parameter describing when the engine is running.
    '''
    engnum = 4
    name = 'Eng (4) Running'

    def derive(self,
               eng_n1=P('Eng (4) N1'),
               eng_n2=P('Eng (4) N2'),
               eng_np=P('Eng (4) Np'),
               fuel_flow=P('Eng (4) Fuel Flow')):
        self.array = self.determine_running(eng_n1, eng_n2, eng_np, fuel_flow)


class Eng_AllRunning(MultistateDerivedParameterNode, EngRunning):
    '''
    Discrete parameter describing when all available engines are running.
    '''
    name = 'Eng (*) All Running'

    @classmethod
    def can_operate(cls, available):
        return 'Eng (*) N1 Min' in available or \
               'Eng (*) N2 Min' in available or \
               'Eng (*) Np Min' in available or \
               'Eng (*) Fuel Flow Min' in available

    def derive(self,
               eng_n1=P('Eng (*) N1 Min'),
               eng_n2=P('Eng (*) N2 Min'),
               eng_np=P('Eng (*) Np Min'),
               fuel_flow=P('Eng (*) Fuel Flow Min')):
        self.array = self.determine_running(eng_n1, eng_n2, eng_np, fuel_flow)


class Eng_AnyRunning(MultistateDerivedParameterNode, EngRunning):
    '''
    Discrete parameter describing when any engines are running.

    This is useful with 'Eng (*) All Running' to detect if not all engines are
    running.
    '''
    name = 'Eng (*) Any Running'
    
    @classmethod
    def can_operate(cls, available):
        return 'Eng (*) N1 Max' in available or \
               'Eng (*) N2 Max' in available or \
               'Eng (*) Np Max' in available or \
               'Eng (*) Fuel Flow Max' in available

    def derive(self,
               eng_n1=P('Eng (*) N1 Max'),
               eng_n2=P('Eng (*) N2 Max'),
               eng_np=P('Eng (*) Np Max'),
               fuel_flow=P('Eng (*) Fuel Flow Max')):
        self.array = self.determine_running(eng_n1, eng_n2, eng_np, fuel_flow)


class ThrustModeSelected(MultistateDerivedParameterNode):
    '''
    Combines Thrust Mode Selected parameters.
    '''

    values_mapping = {
        0: '-',
        1: 'Selected',
    }

    @classmethod
    def can_operate(cls, available):
        return any_of(cls.get_dependency_names(), available)

    def derive(self,
               thrust_l=P('Thrust Mode Selected (L)'),
               thrust_r=P('Thrust Mode Selected (R)')):

        thrusts = [thrust for thrust in [thrust_l,
                                         thrust_r] if thrust]

        if len(thrusts) == 1:
            self.array = thrusts[0].array

        array = MappedArray(np_ma_zeros_like(thrusts[0].array, dtype=np.short),
                            values_mapping=self.values_mapping)

        masks = []
        for thrust in thrusts:
            masks.append(thrust.array.mask)
            array[thrust.array == 'Selected'] = 'Selected'

        array.mask = merge_masks(masks)
        self.array = array


class EventMarker(MultistateDerivedParameterNode):
    '''
    Combine Event Marker from multiple sources where recorded separately.
    '''
    values_mapping = {0: '-', 1: 'Event'}
    name = 'Event Marker'

    @classmethod
    def can_operate(cls, available):
        return any_of(cls.get_dependency_names(), available)

    def derive(self,
               event_marker_1=M('Event Marker (1)'),
               event_marker_2=M('Event Marker (2)'),
               event_marker_3=M('Event Marker (3)'),
               event_marker_capt=M('Event Marker (Capt)'),
               event_marker_fo=M('Event Marker (FO)')):

        self.array = vstack_params_where_state(
            (event_marker_1, 'Event'),
            (event_marker_2, 'Event'),
            (event_marker_3, 'Event'),
            (event_marker_capt, 'Event'),
            (event_marker_fo, 'Event'),
        ).any(axis=0)


class Flap(MultistateDerivedParameterNode):
    '''
    Steps raw Flap angle from surface into detents.
    '''

    units = ut.DEGREE
    # Currently uses the frequency of the Flap Angle parameter - might
    # consider upsampling to 2Hz for the Kernal sizes in the calculate_flap
    # function 
    ##align_frequency = 2

    @classmethod
    def can_operate(cls, available, frame=A('Frame'),
                    model=A('Model'), series=A('Series'), family=A('Family')):

        frame_name = frame.value if frame else None

        if frame_name == 'L382-Hercules':
            return 'Altitude AAL' in available

        if not all_of(('Flap Angle', 'Model', 'Series', 'Family'), available):
            return False

        try:
            at.get_flap_map(model.value, series.value, family.value)
        except KeyError:
            #Q: Everyone should have a flap map - so raise error?
            cls.exception("No flap mapping available for '%s', '%s', '%s'.",
                          model.value, series.value, family.value)
            return False

        return True

    def derive(self, flap=P('Flap Angle'),
               model=A('Model'), series=A('Series'), family=A('Family'),
               frame=A('Frame'), alt_aal=P('Altitude AAL')):

        frame_name = frame.value if frame else None

        if frame_name == 'L382-Hercules':
            self.values_mapping = {0: '0', 50: '50', 100: '100'}

            self.units = ut.PERCENT  # Hercules flaps are unique in this regard!

            # Flap is not recorded, so invent one of the correct length.
            flap_herc = np_ma_zeros_like(alt_aal.array)

            # Takeoff is normally with 50% flap382
            _, toffs = slices_from_to(alt_aal.array, 0.0, 1000.0)
            flap_herc[:toffs[0].stop] = 50.0

            # Assume 50% from 2000 to 1000ft, and 100% thereafter on the approach.
            _, apps = slices_from_to(alt_aal.array, 2000.0, 0.0)
            flap_herc[apps[-1].start:] = np.ma.where(alt_aal.array[apps[-1].start:] > 1000.0, 50.0, 100.0)

            self.array = np.ma.array(flap_herc)
            self.frequency, self.offset = alt_aal.frequency, alt_aal.offset
            return
        
        self.values_mapping, self.array, self.frequency, self.offset = calculate_flap(
            'lever',
            flap,
            model,
            series,
            family,
        )


class FlapLever(MultistateDerivedParameterNode):
    '''
    Rounds the Flap Lever Angle to the selected detent at the start of the
    angle movement.

    Flap is not used to synthesize Flap Lever as this could be misleading.
    Instead, all safety Key Point Values will use Flap Lever followed by Flap
    if Flap Lever is not available.
    '''

    units = ut.DEGREE

    @classmethod
    def can_operate(cls, available,
                    model=A('Model'), series=A('Series'), family=A('Family')):

        if not all_of(('Flap Lever Angle', 'Model', 'Series', 'Family'), available):
            return False

        try:
            at.get_lever_map(model.value, series.value, family.value)
        except KeyError:
            cls.debug("No lever mapping available for '%s', '%s', '%s'.",
                      model.value, series.value, family.value)
            return False

        return True

    def derive(self, flap_lever=P('Flap Lever Angle'),
               model=A('Model'), series=A('Series'), family=A('Family')):

        #self.values_mapping = at.get_lever_map(model.value, series.value, family.value)
        #self.array, self.frequency, self.offset = calculate_surface_angle(
            #'lever',
            #flap_lever,
            #self.values_mapping.keys(),
        #)
        self.values_mapping = at.get_lever_map(model.value, series.value, family.value)
        self.array = step_values(repair_mask(flap_lever.array),
                                 self.values_mapping.keys(),
                                 flap_lever.hz, step_at='move_start')



class FlapIncludingTransition(MultistateDerivedParameterNode):
    '''
    Specifically designed to cater for maintenance monitoring, this assumes
    that when moving the higher of the start and endpoints of the movement
    apply. This increases the chance of needing a flap overspeed inspection,
    but provides a more cautious interpretation of the maintenance
    requirements.
    '''

    units = ut.DEGREE
    
    @classmethod
    def can_operate(cls, available,
                    model=A('Model'), series=A('Series'), family=A('Family')):

        if not all_of(('Flap Angle', 'Model', 'Series', 'Family'), available):
            return False

        try:
            at.get_flap_map(model.value, series.value, family.value)
        except KeyError:
            cls.debug("No lever mapping available for '%s', '%s', '%s'.",
                      model.value, series.value, family.value)
            return False

        return True
    
    def derive(self, flap_angle=P('Flap Angle'),
               model=A('Model'), series=A('Series'), family=A('Family')):
        self.values_mapping = at.get_flap_map(model.value, series.value, family.value)
        self.array = including_transition(flap_angle.array, self.values_mapping)


class FlapExcludingTransition(MultistateDerivedParameterNode):
    '''
    Specifically designed to cater for maintenance monitoring, this assumes
    that when moving the higher of the start and endpoints of the movement
    apply. This increases the chance of needing a flap overspeed inspection,
    but provides a more cautious interpretation of the maintenance
    requirements.
    '''

    units = ut.DEGREE
    
    @classmethod
    def can_operate(cls, available,
                    model=A('Model'), series=A('Series'), family=A('Family')):

        if not all_of(('Flap Angle', 'Model', 'Series', 'Family'), available):
            return False

        try:
            at.get_flap_map(model.value, series.value, family.value)
        except KeyError:
            cls.debug("No lever mapping available for '%s', '%s', '%s'.",
                      model.value, series.value, family.value)
            return False

        return True
    
    def derive(self, flap_angle=P('Flap Angle'),
               model=A('Model'), series=A('Series'), family=A('Family')):
        self.values_mapping, self.array, self.frequency, self.offset = calculate_flap(
            'excluding',
            flap_angle,
            model,
            series,
            family,
        )


class FlapLeverSynthetic(MultistateDerivedParameterNode):
    '''
    Create a synthetic representation of the Flap Lever position.
    '''

    name = 'Flap Lever (Synthetic)'
    units = ut.DEGREE
    align_frequency = 2  # force higher than most Flap frequencies

    @classmethod
    def can_operate(cls, available,
                    model=A('Model'), series=A('Series'), family=A('Family')):

        if not all_of(('Flap', 'Model', 'Series', 'Family'), available):
            return False

        try:
            angles = at.get_conf_angles(model.value, series.value, family.value)
        except KeyError:
            # try lever map if no conf
            try:
                angles = at.get_lever_angles(model.value, series.value, family.value)
            except KeyError:
                cls.warning("No lever angles available for '%s', '%s', '%s'.",
                            model.value, series.value, family.value)
                return False
        
        can_operate = True

        slat_required = any(slat is not None for slat, flap, flaperon in
                            angles.values())
        if slat_required:
            can_operate = can_operate and 'Slat' in available

        flaperon_required = any(flaperon is not None for slat, flap, flaperon in
                                angles.values())
        if flaperon_required:
            can_operate = can_operate and 'Flaperon' in available

        return can_operate

    def derive(self, flap=M('Flap'), slat=M('Slat'), flaperon=M('Flaperon'),
               model=A('Model'), series=A('Series'), family=A('Family')):
        try:
            angles = at.get_conf_angles(model.value, series.value, family.value)
            use_conf = True
        except KeyError:
            angles = at.get_lever_angles(model.value, series.value, family.value)
            use_conf = False

        # Get the values mapping, airbus requires some hacking:
        if use_conf:
            self.values_mapping = at.constants.LEVER_STATES
        else:
            self.values_mapping = at.get_lever_map(model.value, series.value, family.value)

        # Prepare the destination array:
        self.array = MappedArray(np_ma_masked_zeros_like(flap.array),
                                 values_mapping=self.values_mapping)

        # Update the destination array according to the mappings:
        for (state, (s, f, a)) in angles.iteritems():
            condition = (flap.array == str(f))
            if s is not None:
                condition &= (slat.array == str(s))
            if a is not None:
                condition &= (flaperon.array == str(a))
            if use_conf:
                state = at.constants.CONF_TO_LEVER[state]
            self.array[condition] = state

        # Repair the mask to smooth out transitions:
        nearest_neighbour_mask_repair(self.array, copy=False,
                                      repair_gap_size=(30 * self.hz),
                                      direction='backward')


class Flaperon(MultistateDerivedParameterNode):
    '''
    Where Ailerons move together and used as Flaps, these are known as
    "Flaperon" control.

    Flaperons are measured where both Left and Right Ailerons move down,
    which on the left creates possitive roll but on the right causes negative
    roll. The difference of the two signals is the Flaperon control.

    The Flaperon is stepped at the start of movement into the nearest aileron
    detents, e.g. 0, 5, 10 deg

    Note: This is used for Airbus models and does not necessarily mean as
    much to other aircraft types.
    '''

    @classmethod
    def can_operate(cls, available,
                    model=A('Model'), series=A('Series'), family=A('Family')):

        if not all_of(('Aileron (L)', 'Aileron (R)', 'Model', 'Series', 'Family'), available):
            return False

        try:
            at.get_aileron_map(model.value, series.value, family.value)
        except KeyError:
            cls.debug("No aileron/flaperon mapping available for '%s', '%s', '%s'.",
                      model.value, series.value, family.value)
            return False

        return True

    def derive(self, al=P('Aileron (L)'), ar=P('Aileron (R)'),
               model=A('Model'), series=A('Series'), family=A('Family')):
        # Take the difference of the two signals (which should cancel each
        # other out when rolling) and divide the range by two (to account for
        # the left going negative and right going positive when flaperons set)
        #al.array = (al.array - ar.array) / 2

        #self.values_mapping = at.get_aileron_map(model.value, series.value, family.value)
        #self.array, self.frequency, self.offset = calculate_surface_angle(
            #'lever',
            #al,
            #self.values_mapping.keys(),
        #)
        flaperon_angle = (al.array - ar.array) / 2
        self.values_mapping = at.get_aileron_map(model.value, series.value, family.value)
        self.array = step_values(flaperon_angle,
                                 self.values_mapping.keys(),
                                 al.hz, step_at='move_start')


class FuelQty_Low(MultistateDerivedParameterNode):
    '''
    '''
    name = "Fuel Qty (*) Low"
    values_mapping = {
        0: '-',
        1: 'Warning',
    }

    @classmethod
    def can_operate(cls, available):
        return any_of(('Fuel Qty Low', 'Fuel Qty (L) Low', 'Fuel Qty (R) Low'),
                      available)

    def derive(self, fqty=M('Fuel Qty Low'),
               fqty1=M('Fuel Qty (L) Low'),
               fqty2=M('Fuel Qty (R) Low')):
        warning = vstack_params_where_state(
            (fqty, 'Warning'),
            (fqty1, 'Warning'),
            (fqty2, 'Warning'),
        )
        self.array = warning.any(axis=0)


class GearDown(MultistateDerivedParameterNode):
    '''
    This Multi-State parameter uses "majority voting" to decide whether the
    gear is up or down.

    If Gear (*) Down is not recorded, it will be created from Gear Down
    Selected which is from the cockpit lever.

    TODO: Add a transit delay (~10secs) to the selection to when the gear is
    down.
    '''

    align = False
    values_mapping = {
        0: 'Up',
        1: 'Down',
    }

    @classmethod
    def can_operate(cls, available):
        # Can operate with a any combination of parameters available
        gear_downs = ('Gear (L) Down', 'Gear (N) Down', 'Gear (R) Down', 'Gear Down Selected')
        return any_of(gear_downs, available) or all_of(('Gear Up', 'Gear In Transit'), available)

    def derive(self,
               gl=M('Gear (L) Down'),
               gn=M('Gear (N) Down'),
               gr=M('Gear (R) Down'),
               gear_up=M('Gear Up'),
               gear_transit=M('Gear In Transit'),
               gear_sel=M('Gear Down Selected')):
        # Join all available gear parameters and use whichever are available.
        if gl or gn or gr:
            self.array = vstack_params_where_state(
                (gl, 'Down'),
                (gn, 'Down'),
                (gr, 'Down'),
            ).any(axis=0)
        elif gear_up and gear_transit:
            not_up = vstack_params_where_state(
                (gear_up, 'Up'),
                (gear_transit, 'In Transit'),
            ).any(axis=0)
            self.array = ~not_up
        else:  # gear_sel
            self.array = gear_sel.array


class GearUp(MultistateDerivedParameterNode):
    '''
    This Multi-State parameter uses "majority voting" to decide whether the
    gear is up or down.
    '''

    align = False
    values_mapping = {
        0: 'Down',
        1: 'Up',
    }

    @classmethod
    def can_operate(cls, available):
        # Can operate with a any combination of parameters available
        merge_gear_up = any_of(('Gear (L) Up', 'Gear (N) Up', 'Gear (R) Up'), available)
        calc_gear_up = all_of(('Gear Up Selected', 'Gear (*) Red Warning'), available)
        up_sel_transit = all_of(('Gear Up Selected', 'Gear In Transit'), available)
        return merge_gear_up or calc_gear_up or up_sel_transit

    def derive(self,
               gl=M('Gear (L) Up'),
               gn=M('Gear (N) Up'),
               gr=M('Gear (R) Up'),
               gear_up_sel=M('Gear Up Selected'),
               gear_warn=M('Gear (*) Red Warning'),
               gear_transit=M('Gear In Transit')
               ):

        if gl or gn or gr:
            # Join all available gear parameters and use whichever are available.
            self.array = vstack_params_where_state(
                (gl, 'Up'),
                (gn, 'Up'),
                (gr, 'Up'),
            ).any(axis=0)
        else:
            # we need to align the gear down and gear red warnings parameters
            # before we continue
            movement_param = gear_warn if gear_warn else gear_transit
            movement_state = 'Warning' if gear_warn else 'In Transit'
            if gear_up_sel.frequency > movement_param.frequency:
                movement_param = movement_param.get_aligned(gear_up_sel)
            else:
                gear_up_sel = gear_up_sel.get_aligned(movement_param)
            self.frequency = gear_up_sel.frequency
            self.offset = gear_up_sel.offset
            self.array = np.zeros_like(gear_up_sel.array, dtype=np.short)
            self.array[gear_up_sel.array == 'Up'] = 'Up'
            # Calculate gear up from gear down and gear red warnings.
            # We use up to 10s of `Gear (*) Red Warning` == 'Warning'
            # preceeding the actual gear position change state to define the
            # gear transition.
            start_end_changes = find_edges_on_state_change(
                movement_state, movement_param.array, change='entering_and_leaving')
            starts = start_end_changes[::2]
            ends = start_end_changes[1::2]
            start_end_changes = zip(starts, ends)

            for start, end in start_end_changes:
                # for clarity, we're only interested in the end of the
                # transition - so ceiling finds the end
                start = math.ceil(start)
                end = math.ceil(end)
                if (end - start) / self.frequency > 10:
                    # we are only using 10s gear transitions
                    end = start + 10 * self.frequency

                # look for state before gear started moving (back one sample)
                if gear_up_sel.array[start - 1] == 'Down':
                    # Prepend the warning to the gear position to define the
                    # selection
                    self.array[start:end] = 'Down'
            self.array.mask = gear_up_sel.array.mask


class GearInTransit(MultistateDerivedParameterNode):
    '''
    This Multi-State parameter uses "majority voting" to decide whether the
    gear is in transit.
    '''

    align = False
    values_mapping = {
        0: '-',
        1: 'In Transit',
    }

    @classmethod
    def can_operate(cls, available):
        # Can operate with a any combination of parameters available
        return any_of(cls.get_dependency_names(), available)

    def derive(self,
               gl=M('Gear (L) In Transit'),
               gn=M('Gear (N) In Transit'),
               gr=M('Gear (R) In Transit')):

        self.array = vstack_params_where_state(
            (gl, 'In Transit'),
            (gn, 'In Transit'),
            (gr, 'In Transit'),
        ).any(axis=0)


class GearOnGround(MultistateDerivedParameterNode):
    '''
    Combination of left and right main gear signals.
    '''
    align = False
    values_mapping = {
        0: 'Air',
        1: 'Ground',
    }

    @classmethod
    def can_operate(cls, available):
        return any_of(cls.get_dependency_names(), available)

    def derive(self,
               gl=M('Gear (L) On Ground'),
               gr=M('Gear (R) On Ground')):

        # Note that this is not needed on the following frames which record
        # this parameter directly: 737-4, 737-i

        if gl and gr:
            delta = abs((gl.offset - gr.offset) * gl.frequency)
            if 0.75 < delta or delta < 0.25:
                # If the samples of the left and right gear are close together,
                # the best representation is to map them onto a single
                # parameter in which we accept that either wheel on the ground
                # equates to gear on ground.
                self.array = np.ma.logical_or(gl.array, gr.array)
                self.frequency = gl.frequency
                self.offset = gl.offset
                return
            else:
                # If the paramters are not co-located, then
                # merge_two_parameters creates the best combination possible.
                self.array, self.frequency, self.offset = merge_two_parameters(gl, gr)
                return
        if gl:
            gear = gl
        else:
            gear = gr
        self.array = gear.array
        self.frequency = gear.frequency
        self.offset = gear.offset


class GearDownSelected(MultistateDerivedParameterNode):
    '''
    Red warnings are included as the selection may first be indicated by one
    of the red warning lights coming on, rather than the gear status
    changing.

    This is the basis for 'Gear Up Selected'.

    TODO: Derive from "Gear Up" only if recorded.
    '''
    align_frequency = 1

    values_mapping = {
        0: 'Up',
        1: 'Down',
    }

    @classmethod
    def can_operate(cls, available):
        return 'Gear Down' in available

    def derive(self,
               gear_down=M('Gear Down'),
               gear_warn=M('Gear (*) Red Warning')):

        self.array = np.zeros_like(gear_down.array, dtype=np.short)
        self.array[gear_down.array == 'Down'] = 'Down'
        self.array = repair_mask(self.array, method='fill_start')
        if gear_warn:
            # We use up to 10s of `Gear (*) Red Warning` == 'Warning'
            # preceeding the actual gear position change state to define the
            # gear transition.
            start_end_warnings = find_edges_on_state_change(
                'Warning', gear_warn.array, change='entering_and_leaving')
            starts = start_end_warnings[::2]
            ends = start_end_warnings[1::2]
            start_end_warnings = zip(starts, ends)

            for start, end in start_end_warnings:
                # for clarity, we're only interested in the end of the
                # transition - so ceiling finds the end
                start = math.ceil(start)
                end = math.ceil(end)
                if (end - start) / self.frequency > 10:
                    # we are only using 10s gear transitions
                    start = end - 10 * self.frequency

                # look for state before gear started moving (back one sample)
                if gear_down.array[end + 1] == 'Down':
                    # Prepend the warning to the gear position to define the
                    # selection
                    self.array[start:end + 1] = 'Down'


class GearUpSelected(MultistateDerivedParameterNode):
    '''
    This is the inverse of 'Gear Down Selected' which does all the hard work
    for us establishing transitions from 'Gear Down' with the assocaited Red
    Warnings.
    '''
    align_frequency = 1

    values_mapping = {
        0: 'Down',
        1: 'Up',
    }

    def derive(self, gear_dn_sel=M('Gear Down Selected')):
        # Invert the Gear Down Selected array
        self.array = 1 - gear_dn_sel.array.raw


class Gear_RedWarning(MultistateDerivedParameterNode):
    '''
    Merges all the Red Warning systems for Nose, Left and Right gears.
    Ensures that false warnings on the ground are ignored.
    '''
    name = 'Gear (*) Red Warning'
    values_mapping = {0: '-',
                      1: 'Warning'}
    #store in hdf = False! glimpse into the future ;)

    @classmethod
    def can_operate(self, available):
        return 'Airborne' in available and any_of((
            'Gear (L) Red Warning',
            'Gear (N) Red Warning',
            'Gear (R) Red Warning',
        ), available)

    def derive(self,
               gear_warn_l=M('Gear (L) Red Warning'),
               gear_warn_n=M('Gear (N) Red Warning'),
               gear_warn_r=M('Gear (R) Red Warning'),
               airs=S('Airborne')):

        # Join available gear parameters and use whichever are available.
        red_warning = vstack_params_where_state(
            (gear_warn_l, 'Warning'),
            (gear_warn_n, 'Warning'),
            (gear_warn_r, 'Warning'),
        )
        in_air = np.zeros(len(red_warning[0]), dtype=np.bool)
        for air in airs:
            in_air[air.slice] = 1
        # ensure that the red warnings were in the air
        ##gear_warn = M(array=red_warning.any(axis=0), values_mapping={
            ##True: 'Warning'})
        red_air = red_warning.any(axis=0) & in_air
        # creating mapped array is probably not be required due to __setattr__
        red = np.ma.zeros(len(red_air), dtype=np.short)
        red[red_air] = 1
        self.array = MappedArray(red, values_mapping=self.values_mapping)


class ILSInnerMarker(MultistateDerivedParameterNode):
    '''
    Combine ILS Marker for captain and first officer where recorded separately.
    '''
    values_mapping = {0: '-', 1: 'Present'}
    align = False
    name = 'ILS Inner Marker'

    @classmethod
    def can_operate(cls, available):
        return any_of(cls.get_dependency_names(), available)

    def derive(self,
               ils_mkr_capt=M('ILS Inner Marker (Capt)'),
               ils_mkr_fo=M('ILS Inner Marker (FO)')):

        self.array = vstack_params_where_state(
            (ils_mkr_capt, 'Present'),
            (ils_mkr_fo, 'Present'),
        ).any(axis=0)


class ILSMiddleMarker(MultistateDerivedParameterNode):
    '''
    Combine ILS Marker for captain and first officer where recorded separately.
    '''
    values_mapping = {0: '-', 1: 'Present'}
    align = False
    name = 'ILS Middle Marker'

    @classmethod
    def can_operate(cls, available):
        return any_of(cls.get_dependency_names(), available)

    def derive(self,
               ils_mkr_capt=M('ILS Middle Marker (Capt)'),
               ils_mkr_fo=M('ILS Middle Marker (FO)')):

        self.array = vstack_params_where_state(
            (ils_mkr_capt, 'Present'),
            (ils_mkr_fo, 'Present'),
        ).any(axis=0)


class ILSOuterMarker(MultistateDerivedParameterNode):
    '''
    Combine ILS Marker for captain and first officer where recorded separately.
    '''
    values_mapping = {0: '-', 1: 'Present'}
    align = False
    name = 'ILS Outer Marker'

    @classmethod
    def can_operate(cls, available):
        return any_of(cls.get_dependency_names(), available)

    def derive(self,
               ils_mkr_capt=M('ILS Outer Marker (Capt)'),
               ils_mkr_fo=M('ILS Outer Marker (FO)')):

        self.array = vstack_params_where_state(
            (ils_mkr_capt, 'Present'),
            (ils_mkr_fo, 'Present'),
        ).any(axis=0)


class KeyVHFCapt(MultistateDerivedParameterNode):

    name = 'Key VHF (Capt)'
    values_mapping = {0: '-', 1: 'Keyed'}

    @classmethod
    def can_operate(cls, available):
        return any_of(('Key VHF (1) (Capt)',
                       'Key VHF (2) (Capt)',
                       'Key VHF (3) (Capt)'), available)

    def derive(self, key_vhf_1=M('Key VHF (1) (Capt)'),
               key_vhf_2=M('Key VHF (2) (Capt)'),
               key_vhf_3=M('Key VHF (3) (Capt)')):
        self.array = vstack_params_where_state(
            (key_vhf_1, 'Keyed'),
            (key_vhf_2, 'Keyed'),
            (key_vhf_3, 'Keyed'),
        ).any(axis=0)


class KeyVHFFO(MultistateDerivedParameterNode):

    name = 'Key VHF (FO)'
    values_mapping = {0: '-', 1: 'Keyed'}

    @classmethod
    def can_operate(cls, available):
        return any_of(('Key VHF (1) (FO)',
                       'Key VHF (2) (FO)',
                       'Key VHF (3) (FO)'), available)

    def derive(self, key_vhf_1=M('Key VHF (1) (FO)'),
               key_vhf_2=M('Key VHF (2) (FO)'),
               key_vhf_3=M('Key VHF (3) (FO)')):
        self.array = vstack_params_where_state(
            (key_vhf_1, 'Keyed'),
            (key_vhf_2, 'Keyed'),
            (key_vhf_3, 'Keyed'),
        ).any(axis=0)


class MasterCaution(MultistateDerivedParameterNode):
    '''
    Combine Master Caution for captain and first officer.
    '''
    values_mapping = {0: '-', 1: 'Caution'}

    @classmethod
    def can_operate(cls, available):
        return any_of(cls.get_dependency_names(), available)

    def derive(self,
               capt=M('Master Caution (Capt)'),
               fo=M('Master Caution (FO)'),
               capt_2=M('Master Caution (Capt)(2)'),
               fo_2=M('Master Caution (FO)(2)'),
               ):

        self.array = vstack_params_where_state(
            (capt, 'Caution'),
            (fo, 'Caution'),
            (capt_2, 'Caution'),
            (fo_2, 'Caution'),
        ).any(axis=0)


class MasterWarning(MultistateDerivedParameterNode):
    '''
    Combine master warning for captain and first officer.
    '''
    values_mapping = {0: '-', 1: 'Warning'}

    @classmethod
    def can_operate(cls, available):
        return any_of(cls.get_dependency_names(), available)

    def derive(self,
               warn_capt=M('Master Warning (Capt)'),
               warn_fo=M('Master Warning (FO)')):

        self.array = vstack_params_where_state(
            (warn_capt, 'Warning'),
            (warn_fo, 'Warning'),
        ).any(axis=0)


class PackValvesOpen(MultistateDerivedParameterNode):
    '''
    Integer representation of the combined pack configuration.
    '''

    name = 'Pack Valves Open'

    values_mapping = {
        0: 'All closed',
        1: 'One engine low flow',
        2: 'Flow level 2',
        3: 'Flow level 3',
        4: 'Both engines high flow',
    }

    @classmethod
    def can_operate(cls, available):
        '''
        '''
        # Works with both 'ECS Pack (1) On' and 'ECS Pack (2) On' ECS Pack High Flows are optional
        return all_of(['ECS Pack (1) On', 'ECS Pack (2) On'], available)

    def derive(self,
               p1=M('ECS Pack (1) On'), p1h=M('ECS Pack (1) High Flow'),
               p2=M('ECS Pack (2) On'), p2h=M('ECS Pack (2) High Flow')):
        '''
        '''
        # TODO: account properly for states/frame specific fixes
        # Sum the open engines, allowing 1 for low flow and 1+1 for high flow
        # each side.
        flow = p1.array.raw + p2.array.raw
        if p1h and p2h:
            flow = p1.array.raw * (1 + p1h.array.raw) + p2.array.raw * (1 + p2h.array.raw)
        self.array = flow
        self.offset = offset_select('mean', [p1, p1h, p2, p2h])


class PilotFlying(MultistateDerivedParameterNode):
    '''
    Determines the pilot flying the aircraft.

    For Airbus aircraft we use the captain and first officer sidestick angles
    to determine who is providing input to the aircraft.

    Reference was made to the following documentation to assist with the
    development of this algorithm:

    - A320 Flight Profile Specification
    - A321 Flight Profile Specification
    '''
    values_mapping = {0: '-', 1: 'Captain', 2: 'First Officer'}

    def derive(self,
               stick_capt=P('Sidestick Angle (Capt)'),
               stick_fo=P('Sidestick Angle (FO)')):


        pilot_flying = MappedArray(np_ma_masked_zeros_like(stick_capt.array, dtype=np.short),
                                   values_mapping=self.values_mapping)

        if stick_capt.array.size > 61:
            # Calculate average instead of sum as it we already have a function
            # defined to work over a window and it doesn't affect the result as
            # the arrays are altered in the same way and are still comparable.
            window = 31 * self.hz  # Use 61 seconds for 30 seconds either side.
            if not window % 2:
                window += 1
            angle_capt = moving_average(np.ma.abs(stick_capt.array), window)
            angle_fo = moving_average(np.ma.abs(stick_fo.array), window)
            # Repair the array as the moving average is padded with masked
            # zeros
            angle_capt = repair_mask(angle_capt, repair_duration=31,
                                     extrapolate=True)
            angle_fo = repair_mask(angle_fo, repair_duration=31,
                                   extrapolate=True)
            # ignore moving average if no input from pilot at that time.
            # AFPS declares 0.5 degrees minimum input, but due to A330/A340 
            # poor resolution, allow 1.7 degrees of movement.
            angle_capt_zerod = np.ma.where(stick_capt.array < 1.7, 0.0, angle_capt)
            angle_fo_zerod = np.ma.where(stick_fo.array < 1.7, 0.0, angle_fo)
            # mask non inputs to allow us to repair nearest neightbour later
            angle_capt_masked = np.ma.masked_where((stick_capt.array == 0.0) & (angle_capt_zerod != 0.0), angle_capt_zerod)
            angle_fo_masked = np.ma.masked_where((stick_fo.array == 0.0) & (angle_fo_zerod != 0.0), angle_fo_zerod)

            pilot_flying[angle_capt_masked > angle_fo_masked] = 'Captain'
            pilot_flying[angle_capt_masked < angle_fo_masked] = 'First Officer'
            # keep calculated masks
            pilot_flying.mask = angle_capt_masked.mask & angle_fo_masked.mask

            # repair nearest neighbour to remove small gaps of no movement
            pilot_flying = nearest_neighbour_mask_repair(pilot_flying, repair_gap_size=20*self.frequency, copy=False)
            # use second window to remove spiking between captain and first
            # officer during dual stick periods
            pilot_flying = second_window(pilot_flying, self.frequency, 2).astype(np.short)

        self.array = pilot_flying


class PitchAlternateLaw(MultistateDerivedParameterNode):
    '''
    Combine Pitch Alternate Law from sources (1) and/or (2).
    '''

    values_mapping = {0: '-', 1: 'Engaged'}

    @classmethod
    def can_operate(cls, available):

        return any_of(cls.get_dependency_names(), available)

    def derive(self,
               alt_law_1=M('Pitch Alternate Law (1)'),
               alt_law_2=M('Pitch Alternate Law (2)')):

        self.array = vstack_params_where_state(
            (alt_law_1, 'Engaged'),
            (alt_law_2, 'Engaged'),
        ).any(axis=0)


class Slat(MultistateDerivedParameterNode):
    '''
    Steps raw slat angle into detents.
    '''

    units = ut.DEGREE

    @classmethod
    def can_operate(cls, available,
                    model=A('Model'), series=A('Series'), family=A('Family')):

        if not all_of(('Slat Angle', 'Model', 'Series', 'Family'), available):
            return False

        try:
            at.get_slat_map(model.value, series.value, family.value)
        except KeyError:
            cls.debug("No slat mapping available for '%s', '%s', '%s'.",
                      model.value, series.value, family.value)
            return False

        return True

    def derive(self, slat=P('Slat Angle'),
               model=A('Model'), series=A('Series'), family=A('Family')):
        self.values_mapping, self.array, self.frequency, self.offset = calculate_slat(
            'lever',
            slat,
            model,
            series,
            family,
        )


class SlatExcludingTransition(MultistateDerivedParameterNode):
    '''
    Specifically designed to cater for maintenance monitoring, this assumes
    that when moving the lower of the start and endpoints of the movement
    apply. This minimises the chance of needing a slat overspeed inspection.
    '''

    units = ut.DEGREE

    @classmethod
    def can_operate(cls, available,
                    model=A('Model'), series=A('Series'), family=A('Family')):

        if not all_of(('Slat Angle', 'Model', 'Series', 'Family'), available):
            return False

        try:
            at.get_slat_map(model.value, series.value, family.value)
        except KeyError:
            cls.debug("No slat mapping available for '%s', '%s', '%s'.",
                      model.value, series.value, family.value)
            return False

        return True

    def derive(self, slat=P('Slat Angle'),
               model=A('Model'), series=A('Series'), family=A('Family')):

        self.values_mapping, self.array, self.frequency, self.offset = calculate_slat(
            'excluding',
            slat,
            model,
            series,
            family,
        )


class SlatIncludingTransition(MultistateDerivedParameterNode):
    '''
    Specifically designed to cater for maintenance monitoring, this assumes
    that when moving the higher of the start and endpoints of the movement
    apply. This increases the chance of needing a slat overspeed inspection,
    but provides a more cautious interpretation of the maintenance
    requirements.
    '''

    units = ut.DEGREE

    @classmethod
    def can_operate(cls, available,
                    model=A('Model'), series=A('Series'), family=A('Family')):

        if not all_of(('Slat Angle', 'Model', 'Series', 'Family'), available):
            return False

        try:
            at.get_slat_map(model.value, series.value, family.value)
        except KeyError:
            cls.debug("No slat mapping available for '%s', '%s', '%s'.",
                      model.value, series.value, family.value)
            return False

        return True

    def derive(self, slat=P('Slat Angle'),
               model=A('Model'), series=A('Series'), family=A('Family')):

        self.values_mapping, self.array, self.frequency, self.offset = calculate_slat(
            'including',
            slat,
            model,
            series,
            family,
        )


class StickPusher(MultistateDerivedParameterNode):
    '''
    Where two Stick Pusher systems are recorded the results are OR'd to make
    a single parameter which operates in response to either system
    triggering.
    '''

    values_mapping = {
        0: '-',
        1: 'Push'
    }

    @classmethod
    def can_operate(cls, available):
        return any_of(('Stick Pusher (L)', 'Stick Pusher (R)'), available)

    def derive(self, spl=M('Stick Pusher (L)'),
               spr=M('Stick Pusher (R)')):

        available = [par for par in [spl, spr] if par]

        if len(available) > 1:
            shake_stack = vstack_params_where_state(*[(s, 'Push') for s in available])
            self.array = shake_stack.any(axis=0)
        elif len(available) == 1:
            self.array = available[0].array


class StickShaker(MultistateDerivedParameterNode):
    '''
    This accounts for the different types of stick shaker system. Where two
    systems are recorded the results are OR'd to make a single parameter which
    operates in response to either system triggering.
    '''

    values_mapping = {
        0: '-',
        1: 'Shake',
    }

    @classmethod
    def can_operate(cls, available):
        return any_of((
            'Stick Shaker (L)',
            'Stick Shaker (R)',
            'Stick Shaker (1)',
            'Stick Shaker (2)',
            'Stick Shaker (3)',
            'Stick Shaker (4)',
        ), available)

    def derive(self, ssl=M('Stick Shaker (L)'),
               ssr=M('Stick Shaker (R)'),
               ss1=M('Stick Shaker (1)'),
               ss2=M('Stick Shaker (2)'),
               ss3=M('Stick Shaker (3)'),
               ss4=M('Stick Shaker (4)'),
               frame=A('Frame'),
               ):

        if frame and frame.value == 'B777':
            #Provision has been included for Boeing 777 type, but until this has been
            #evaluated in detail it raises an exception because there are two bits per
            #shaker, and their operation is not obvious from the documentation.
            raise ValueError

        available = [par for par in [ssl, ssr, ss1, ss2, ss3, ss4,
                                     #b777_L1, b777_L2, b777_R1, b777_R2,
                                     ] if par]
        if len(available) > 1:
            shake_stack = vstack_params_where_state(*[(s, 'Shake') for s in available])
            self.array = shake_stack.any(axis=0)
        elif len(available) == 1:
            self.array = available[0].array


class StallWarning(MultistateDerivedParameterNode):
    '''
    This accounts for the different types of stall warning system. Where two
    systems are recorded the results are OR'd to make a single parameter which
    operates in response to either system triggering.
    '''

    values_mapping = {
        0: '-',
        1: 'Warning',
    }

    @classmethod
    def can_operate(cls, available):
        return any_of((
            'Stall Warning (1)',
            'Stall Warning (2)',
        ), available)

    def derive(self, 
               ss1=M('Stall Warning (1)'),
               ss2=M('Stall Warning (2)'),
               frame=A('Frame'),
               ):

        available = [par for par in [ss1, ss2] if par]
        if len(available) > 1:
            shake_stack = vstack_params_where_state(*[(s, 'Warning') for s in available])
            self.array = shake_stack.any(axis=0)
        elif len(available) == 1:
            self.array = available[0].array


class SpeedbrakeDeployed(MultistateDerivedParameterNode):
    '''
    '''
    values_mapping = {
        0: '-',
        1: 'Deployed',
    }

    @classmethod
    def can_operate(cls, available):
        simple = ('Spoiler (L) Deployed', 'Spoiler (R) Deployed')
        in_out = ('Spoiler (L) Outboard Deployed',
                  'Spoiler (R) Outboard Deployed')
        return all_of(simple, available) or all_of(in_out, available)

    def derive(self, deployed_l=M('Spoiler (L) Deployed'),
               deployed_r=M('Spoiler (R) Deployed'),
               deployed_l_out=M('Spoiler (L) Outboard Deployed'),
               deployed_r_out=M('Spoiler (R) Outboard Deployed')):

        deployed_params = (deployed_l, deployed_r, deployed_l_out, deployed_r_out)
        deployed_stack = vstack_params_where_state(*[(d, 'Deployed') for d in deployed_params])

        array = np_ma_zeros_like(deployed_stack[0], dtype=np.short)
        array = np.ma.where(deployed_stack.all(axis=0), 1, array)

        # mask indexes with greater than 50% masked values
        mask = np.ma.where(deployed_stack.mask.sum(axis=0).astype(float) / len(deployed_stack) * 100 > 50, 1, 0)
        self.array = array
        self.array.mask = mask


class SpeedbrakeSelected(MultistateDerivedParameterNode):
    '''
    Determines the selected state of the speedbrake.

    Speedbrake Selected Values:

    - 0 -- Stowed
    - 1 -- Armed / Commanded (Spoilers Down)
    - 2 -- Deployed / Commanded (Spoilers Up)
    '''

    values_mapping = {
        0: 'Stowed',
        1: 'Armed/Cmd Dn',
        2: 'Deployed/Cmd Up',
    }

    @classmethod
    def can_operate(cls, available, family=A('Family')):
        '''
        '''
        x = available
        if family and family.value == 'BD-100':
            return 'Speedbrake Handle' in x and 'Spoiler Ground Armed' in x
        elif family and family.value == 'Global':
            return any_of(('Speedbrake', 'Speedbrake Handle'), available)
        elif family and family.value in ('CRJ 100/200', 'B777'):
            return 'Speedbrake Handle' in x
        elif family and family.value in ('A318', 'A319', 'A320', 'A321'):
            return 'Speedbrake' in x and 'Speedbrake Armed' in x
        else:
            return ('Speedbrake Deployed' in x or
                    ('Family' in x and 'Speedbrake Switch' in x) or
                    ('Family' in x and 'Speedbrake Handle' in x) or
                    ('Family' in x and 'Speedbrake' in x))

    @classmethod
    def derive_from_handle(cls, handle_array, deployed=1, armed=None,
                           mask_below_armed=False):
        '''
        Basic Speedbrake algorithm for Stowed and Deployed states from
        Spoiler Handle.
        '''
        array = MappedArray(np_ma_masked_zeros_like(handle_array, dtype=np.short),
                            values_mapping=cls.values_mapping)
        stowed_value = deployed
        if armed is not None:
            stowed_value = armed
            array[(handle_array >= stowed_value) & (handle_array < deployed)] = 'Armed/Cmd Dn'
        if mask_below_armed:
            array[handle_array == stowed_value] = 'Stowed'
            array[handle_array < stowed_value] = np.ma.masked
        else:
            array[handle_array < stowed_value] = 'Stowed'
        array[handle_array >= deployed] = 'Deployed/Cmd Up'
        return array

    @staticmethod
    def a320_speedbrake(armed, spdbrk):
        '''
        Speedbrake operation for A320 family.
        '''
        array = np.ma.where(spdbrk.array > 1.0,
                            'Deployed/Cmd Up', armed.array)
        return array

    @staticmethod
    def b737_speedbrake(spdbrk, handle):
        '''
        Speedbrake Handle Positions for 737-x:

            ========    ============
            Angle       Notes
            ========    ============
             0.0        Full Forward
             4.0        Armed
            24.0
            29.0
            38.0        In Flight
            40.0        Straight Up
            48.0        Full Up
            ========    ============

        Speedbrake Positions > 1 = Deployed
        '''
        if spdbrk and handle:
            # Speedbrake and Speedbrake Hbd100_speedbrakeandle available
            '''
            Speedbrake status taken from surface position. This allows
            for aircraft where the handle is inoperative, overwriting
            whatever the handle position is when the brakes themselves
            have deployed.

            It's not possible to identify when the speedbrakes are just
            armed in this case, so we take any significant motion as
            deployed.

            If there is no handle position recorded, the default 'Stowed'
            value is retained.
            '''
            armed = np.ma.where((2.0 < handle.array) & (handle.array < 35.0),
                                'Armed/Cmd Dn', 'Stowed')
            array = np.ma.where((handle.array >= 35.0) | (spdbrk.array > 1.0),
                                'Deployed/Cmd Up', armed)
        elif spdbrk and not handle:
            # Speedbrake only
            array = np.ma.where(spdbrk.array > 1.0,
                                'Deployed/Cmd Up', 'Stowed')
        elif handle and not spdbrk:
            # Speedbrake Handle only
            armed = np.ma.where((2.0 < handle.array) & (handle.array < 35.0),
                                'Armed/Cmd Dn', 'Stowed')
            array = np.ma.where(handle.array >= 35.0,
                                'Deployed/Cmd Up', armed)
        else:
            raise ValueError("Can't work without either Speedbrake or Handle")
        return array

    @classmethod
    def bd100_speedbrake(cls, handle_array, spoiler_gnd_armed_array):
        '''
        Speedbrake Handle non-zero is deployed, Spoiler Ground Armed is
        armed.
        '''
        # Default is stowed.
        array = MappedArray(np_ma_masked_zeros_like(handle_array, dtype=np.short),
                            values_mapping=cls.values_mapping)
        array[spoiler_gnd_armed_array != 'Armed'] = 'Stowed'
        array[spoiler_gnd_armed_array == 'Armed'] = 'Armed/Cmd Dn'
        array[handle_array >= 1] = 'Deployed/Cmd Up'
        return array

    @staticmethod
    def b787_speedbrake(handle):
        '''
        Speedbrake Handle Positions for 787, taken from early recordings.
        '''
        # Speedbrake Handle only
        speedbrake = np.ma.zeros(len(handle.array), dtype=np.short)
        stepped_array = step_values(handle.array, [0, 10, 20])
        # Assuming all values from 15 and above are Deployed. Typically a
        # maximum value of 60 is recorded when deployed with reverse thrust
        # whereas values of 30-60 are seen during the approach.
        speedbrake[stepped_array == 10] = 1
        speedbrake[stepped_array == 20] = 2
        return speedbrake

    @staticmethod
    def learjet_speedbrake(spdsw):
        '''
        Learjet 60XS has a switch with settings:
        0 = Retract
        4 = Extended
        7 = Armed
        6 = Partial

        Here we map thus:
            Retract = Stowed
            Armed = Armed/Cmd Dn
            Partial or Extended = Deployed/Cmd Up
        '''
        switch = spdsw.array
        speedbrake = np.ma.zeros(len(switch), dtype=np.short)
        speedbrake = np.ma.where(switch == 'Retract', 'Stowed',
                                 'Deployed/Cmd Up')
        speedbrake = np.ma.where(switch == 'Armed', 'Armed/Cmd Dn',
                                 speedbrake)
        return speedbrake

    def derive(self,
               deployed=M('Speedbrake Deployed'),
               armed=M('Speedbrake Armed'),
               handle=P('Speedbrake Handle'),
               spdbrk=P('Speedbrake'),
               spdsw=M('Speedbrake Switch'),
               spoiler_gnd_armed=M('Spoiler Ground Armed'),
               family=A('Family')):

        family_name = family.value if family else ''

        if deployed:
            # Families include: A340, ...

            # We have a speedbrake deployed discrete. Set initial state to
            # stowed, then set armed states if available, and finally set
            # deployed state:
            array = np.ma.zeros(len(deployed.array), dtype=np.short)
            if armed:
                array[armed.array == 'Armed'] = 1
            array[deployed.array == 'Deployed'] = 2
            self.array = array

        elif 'B737' in family_name:
            self.array = self.b737_speedbrake(spdbrk, handle)

        elif family_name == 'B747':
            self.array = self.derive_from_handle(handle.array, deployed=5,
                                                 armed=1)

        elif family_name == 'B757':
            self.array = self.derive_from_handle(handle.array, deployed=25,
                                                 armed=12)

        elif family_name == 'B767':
            self.array = self.derive_from_handle(handle.array, deployed=45,
                                                 armed=12)

        elif family_name == 'B777':
            self.array = self.derive_from_handle(handle.array, deployed=10,
                                                 armed=0, mask_below_armed=True)

        elif family_name == 'B787':
            self.array = self.b787_speedbrake(handle)

        elif family_name == 'A300' and not spdbrk:
            # Have only seen Speedbrake Handle ,not Speedbrake parameter so
            # far for A300
            self.array = self.derive_from_handle(handle.array, deployed=10)

        elif family_name in ('A318', 'A319', 'A320', 'A321'):
            self.array = self.a320_speedbrake(armed, spdbrk)

        elif family_name == 'Learjet':
            self.array = self.learjet_speedbrake(spdsw)

        elif family_name == 'G-IV' and spdbrk and handle:
            # based on data seen for G450, clean handle signal with no armed position.
            self.array = np.ma.where((handle.array >= 1.0) | (spdbrk.array > 25.0),
                                'Deployed/Cmd Up', 'Stowed')
        elif family_name in ['G-IV',
                             'G-V',
                             'Global',
                             'CL-600',
                             'BAE 146',
                             'ERJ-170/175',
                             'ERJ-190/195',
                             'Phenom 300'] and spdbrk:
            array = np.ma.zeros(len(spdbrk.array), dtype=np.short)
            if armed:
                # G550 seen with recorded Speedbrake Armed parameter
                array[armed.array == 'Armed'] = 1
            # On the test aircraft SE-RDY the Speedbrake stored 0 at all
            # times and Speedbrake Handle was unresponsive with small numeric
            # variation. The Speedbrake (L) & (R) responded normally so we
            # simply accept over 30deg as deployed.
            self.array = np.ma.where(spdbrk.array > 2.0,
                                     'Deployed/Cmd Up',
                                     array)
        elif family_name in ['ERJ-170/175', 'ERJ-190/195'] and handle:
            self.array = np.ma.where(handle.array < -15.0,
                                     'Stowed',
                                     'Deployed/Cmd Up')

        elif family_name in ['Global', 'CRJ 100/200', 'ERJ-135/145',
                             'CL-600', 'G-IV'] and handle:
            # No valid data seen for this type to date....
            logger.warning(
                'SpeedbrakeSelected: algorithm for family `%s` is undecided, '
                'temporarily using speedbrake handle.', family_name)
            self.array = np_ma_masked_zeros_like(handle.array, dtype=np.short)

        elif family_name == 'BD-100':
            self.array = self.bd100_speedbrake(handle.array,
                                               spoiler_gnd_armed.array)
        else:
            raise NotImplementedError("No Speedbrake mapping for '%s'" % family_name)


class StableApproach(MultistateDerivedParameterNode):
    '''
    During the Descent and Approach, the following steps are assessed in turn
    to determine the aircraft stability:

    1. Gear is down
    2. Landing Flap is set
    3. Track is aligned to Runway (within 12 degrees or 30 if offset approach)
    4. Airspeed minus selected approach speed within -5 to +10 knots (for 3 secs)
    5. Glideslope deviation within 1 dot
    6. Localizer deviation within 1 dot
    7. Vertical speed between -1100 and -200 fpm
    8. Engine Thurst greater than 45% N1 or 35% (A319/B787) or 1.09 EPR (for 10 secs)

    if all the above steps are met, the result is the declaration of:
    9. "Stable"

    Notes:

    Airspeed is relative to "Airspeed Selected" where available as this will
    account for the reference speed and any compensation for the wind speed.

    If Vapp is recorded, a more constraint airspeed threshold is applied.

    Where parameters are not monitored below a certain threshold (e.g. ILS
    below 200ft) the stability criteria just before 200ft is reached is
    continued through to landing. So if one was unstable due to ILS
    Glideslope down to 200ft, that stability is assumed to continue through
    to landing.

    TODO/REVIEW:
    ============
    * Check for 300ft limit if turning onto runway late and ignore stability
      criteria before this? Alternatively only assess criteria when heading is 
      within 50.
    * Add hysteresis (3 second gliding windows for GS / LOC etc.)
    * Engine cycling check
    * Use Engine TPR for B787 instead of EPR if available.
    '''

    values_mapping = {
        0: '-',  # All values should be masked anyway, this helps align values
        1: 'Gear Not Down',
        2: 'Not Landing Flap',
        3: 'Track Not Aligned',
        4: 'Aspd Not Stable',  # Q: Split into two Airspeed High/Low?
        5: 'GS Not Stable',
        6: 'Loc Not Stable',
        7: 'IVV Not Stable',
        8: 'Eng Thrust Not Stable',
        9: 'Stable',
    }

    align_frequency = 1  # force to 1Hz

    @classmethod
    def can_operate(cls, available):
        # Many parameters are optional dependencies
        deps = ['Approach Information', 'Descent',
                'Gear Down', 'Flap',
                'Track Deviation From Runway',
                'Vertical Speed',
                'Altitude AAL',
                ]
        return all_of(deps, available) and (
            'Eng (*) N1 Avg For 10 Sec' in available or
            'Eng (*) EPR Avg For 10 Sec' in available)

    def derive(self,
               apps=A('Approach Information'),
               phases=S('Descent'),
               gear=M('Gear Down'),
               flap=M('Flap'),
               tdev=P('Track Deviation From Runway'),
               aspd_rel=P('Airspeed Relative For 3 Sec'),
               aspd_minus_sel=P('Airspeed Minus Airspeed Selected For 3 Sec'),
               vspd=P('Vertical Speed'),
               gdev=P('ILS Glideslope'),
               ldev=P('ILS Localizer'),
               eng_n1=P('Eng (*) N1 Avg For 10 Sec'),
               eng_epr=P('Eng (*) EPR Avg For 10 Sec'),
               alt=P('Altitude AAL'),
               vapp=P('Vapp'),
               family=A('Family')):

        # create an empty fully masked array
        self.array = np.ma.zeros(len(alt.array), dtype=np.short)
        self.array.mask = True
        # shortcut for repairing masks
        repair = lambda ar, ap, method='interpolate': repair_mask(
            ar[ap], raise_entirely_masked=False, method=method)

        for approach, phase in zip(apps, phases):
            # use Combined descent phase slice as it contains the data from
            # top of descent to touchdown (approach starts and finishes later)
            approach.slice = phase.slice

            # FIXME: approaches shorter than 10 samples will not work due to
            # the use of moving_average with a width of 10 samples.
            if approach.slice.stop - approach.slice.start < 10:
                continue
            # Restrict slice to 10 seconds after landing if we hit the ground
            gnd = index_at_value(alt.array, 0, approach.slice)
            if gnd and gnd + 10 < approach.slice.stop:
                stop = gnd + 10
            else:
                stop = approach.slice.stop
            _slice = slice(approach.slice.start, stop)
            # prepare data for this appproach:
            gear_down = repair(gear.array, _slice, method='fill_start')
            flap_lever = repair(flap.array, _slice, method='fill_start')
            track_dev = repair(tdev.array, _slice)
            if aspd_minus_sel:
                airspeed = repair(aspd_minus_sel.array, _slice)
            elif aspd_rel:
                airspeed = repair(aspd_rel.array, _slice)
            else:
                airspeed = None
            glideslope = repair(gdev.array, _slice) if gdev else None  # optional
            localizer = repair(ldev.array, _slice) if ldev else None  # optional
            # apply quite a large moving average to smooth over peaks and troughs
            vertical_speed = moving_average(repair(vspd.array, _slice), 11)
            if eng_epr:
                # use EPR if available
                engine = repair(eng_epr.array, _slice)
            else:
                engine = repair(eng_n1.array, _slice)
            altitude = repair(alt.array, _slice)

            index_at_50 = index_closest_value(altitude, 50)
            index_at_200 = index_closest_value(altitude, 200)

            #== 1. Gear Down ==
            # Assume unstable due to Gear Down at first
            self.array[_slice] = 1
            landing_gear_set = (gear_down == 'Down')
            stable = landing_gear_set.filled(True)  # assume stable (gear down)

            #== 2. Landing Flap ==
            # not due to landing gear so try to prove it wasn't due to Landing Flap
            self.array[_slice][stable] = 2
            # look for maximum flap used in approach below 1,000ft, otherwise
            # go-arounds can detect the start of flap retracting as the
            # landing flap.
            landing_flap = np.ma.where(altitude < 1000, flap_lever, np.ma.masked).max()
            if landing_flap is np.ma.masked:
                # try looking above 1000ft
                landing_flap = np.ma.where(altitude > 1000, flap_lever, np.ma.masked).max()

            if landing_flap is not np.ma.masked:
                landing_flap_set = (flap_lever == landing_flap)
                # assume stable (flap set)
                stable &= landing_flap_set.filled(True)
            else:
                # All landing flap is masked, assume stable
                logger.warning(
                    'StableApproach: the landing flap is all masked in '
                    'the approach.')
                stable &= True

            #== 3. Track Deviation ==
            self.array[_slice][stable] = 3

            runway = approach.runway
            if runway and runway.get('localizer', {}).get('is_offset'):
                # offset ILS Localizer or offset approach without ILS (IAN approach)
                STABLE_TRACK = 30  # degrees
            else:
                # use 12 to allow rolling a little over the 10 degrees when
                # aligning to runway.
                STABLE_TRACK = 12  # degrees
            stable_track_dev = abs(track_dev) <= STABLE_TRACK
            stable &= stable_track_dev.filled(True)  # assume stable (on track)

            if airspeed is not None:
                #== 4. Airspeed Relative ==
                self.array[_slice][stable] = 4
                if aspd_minus_sel:
                    # Airspeed relative to selected speed
                    STABLE_AIRSPEED_BELOW_REF = -5
                    STABLE_AIRSPEED_ABOVE_REF = 15
                elif vapp:
                    # Those aircraft which record a variable Vapp shall have more constraint thresholds
                    STABLE_AIRSPEED_BELOW_REF = -5
                    STABLE_AIRSPEED_ABOVE_REF = 10
                else:
                    # Most aircraft record only Vref - as we don't know the wind correction be more lenient
                    STABLE_AIRSPEED_BELOW_REF = -5
                    STABLE_AIRSPEED_ABOVE_REF = 35
                stable_airspeed = (airspeed >= STABLE_AIRSPEED_BELOW_REF) & (airspeed <= STABLE_AIRSPEED_ABOVE_REF)
                # extend the stability at the end of the altitude threshold through to landing
                stable_airspeed[altitude < 50] = stable_airspeed[index_at_50]
                stable &= stable_airspeed.filled(True)  # if no V Ref speed, values are masked so consider stable as one is not flying to the vref speed??

            if approach.gs_est:
                #== 5. Glideslope Deviation ==
                self.array[_slice][stable] = 5
                STABLE_GLIDESLOPE = 1.0  # dots
                stable_gs = (abs(glideslope) <= STABLE_GLIDESLOPE)
                # extend the stability at the end of the altitude threshold through to landing
                stable_gs[altitude < 200] = stable_gs[index_at_200]
                stable &= stable_gs.filled(False)  # masked values are usually because they are way outside of range and short spikes will have been repaired

            if approach.gs_est and approach.loc_est:
                #== 6. Localizer Deviation ==
                self.array[_slice][stable] = 6
                STABLE_LOCALIZER = 1.0  # dots
                stable_loc = (abs(localizer) <= STABLE_LOCALIZER)
                # extend the stability at the end of the altitude threshold through to landing
                stable_loc[altitude < 200] = stable_loc[index_at_200]
                stable &= stable_loc.filled(False)  # masked values are usually because they are way outside of range and short spikes will have been repaired

            #== 7. Vertical Speed ==
            self.array[_slice][stable] = 7
            STABLE_VERTICAL_SPEED_MAX = -200
            STABLE_VERTICAL_SPEED_MIN = -1100
            if runway:
                gs_angle = runway.get('glideslope', {}).get('angle')
                # offset ILS Localizer or offset approach without ILS (IAN approach)
                if gs_angle > 3:
                    STABLE_VERTICAL_SPEED_MIN = -1500
            stable_vert = (vertical_speed >= STABLE_VERTICAL_SPEED_MIN) & (vertical_speed <= STABLE_VERTICAL_SPEED_MAX)
            # extend the stability at the end of the altitude threshold through to landing
            stable_vert[altitude < 50] = stable_vert[index_at_50]
            stable &= stable_vert.filled(True)

            #== 8. Engine Thrust (N1/EPR) ==
            self.array[_slice][stable] = 8
            # Patch this value depending upon aircraft type
            if eng_epr:
                if family and family.value in ('A319', 'A320', 'A321'):
                    STABLE_EPR_MIN = 1.02  # Ratio
                else:
                    STABLE_EPR_MIN = 1.09  # Ratio
                stable_engine = (engine >= STABLE_EPR_MIN)
            else:
                if family and family.value in ('B787', 'A319'):
                    STABLE_N1_MIN = 35  # %
                else:
                    STABLE_N1_MIN = 40  # %
                stable_engine = (engine >= STABLE_N1_MIN)
            # extend the stability at the end of the altitude threshold through to landing
            stable_engine[altitude < 50] = stable_engine[index_at_50]
            stable &= stable_engine.filled(True)

            #== 9. Stable ==
            # Congratulations; whatever remains in this approach is stable!
            self.array[_slice][stable] = 9

        #endfor
        return


"""
class StickShaker(MultistateDerivedParameterNode):
    '''
    This accounts for the different types of stick shaker system. Where two
    systems are recorded the results are ORed to make a single parameter which
    operates in response to either system triggering. Hence the removal of
    automatic alignment of the signals.
    '''

    align = False
    values_mapping = {
        0: '-',
        1: 'Shake',
    }

    @classmethod
    def can_operate(cls, available):
        return ('Stick Shaker (L)' in available or \
                'Shaker Activation' in available)

    def derive(self, shake_l=M('Stick Shaker (L)'),
               shake_r=M('Stick Shaker (R)'),
               shake_act=M('Shaker Activation')):
        if shake_l and shake_r:
            self.array = np.ma.logical_or(shake_l.array, shake_r.array)
            self.frequency , self.offset = shake_l.frequency, shake_l.offset

        elif shake_l:
            # Named (L) but in fact (L) and (R) are or'd together at the DAU.
            self.array, self.frequency, self.offset = \
                shake_l.array, shake_l.frequency, shake_l.offset

        elif shake_act:
            self.array, self.frequency, self.offset = \
                shake_act.array, shake_act.frequency, shake_act.offset

        else:
            raise NotImplementedError
"""


class ThrustReversers(MultistateDerivedParameterNode):
    '''
    A single parameter with multi-state mapping as below.
    '''

    # We are interested in all stowed, all deployed or any other combination.
    # The mapping "In Transit" is used for anything other than the fully
    # established conditions, so for example one locked and the other not is
    # still treated as in transit.
    values_mapping = {
        0: 'Stowed',
        1: 'In Transit',
        2: 'Deployed',
    }

    @classmethod
    def can_operate(cls, available):
        return all_of((
            'Eng (1) Thrust Reverser (L) Deployed',
            #'Eng (1) Thrust Reverser (L) Unlocked',   # bonus if available!
            'Eng (1) Thrust Reverser (R) Deployed',
            #'Eng (1) Thrust Reverser (R) Unlocked',   # bonus if available!
            'Eng (2) Thrust Reverser (L) Deployed',
            #'Eng (2) Thrust Reverser (L) Unlocked',   # bonus if available!
            'Eng (2) Thrust Reverser (R) Deployed',
            #'Eng (2) Thrust Reverser (R) Unlocked',   # bonus if available!
        ), available) or all_of((
            #'Eng (1) Thrust Reverser Unlocked',   # bonus if available!
            'Eng (1) Thrust Reverser Deployed',
            #'Eng (2) Thrust Reverser Unlocked',  # bonus if available!
            'Eng (2) Thrust Reverser Deployed',
        ), available) or all_of((
            'Eng (1) Thrust Reverser In Transit',
            'Eng (1) Thrust Reverser Deployed',
            'Eng (2) Thrust Reverser In Transit',
            'Eng (2) Thrust Reverser Deployed',
        ), available) or all_of((
            'Eng (1) Thrust Reverser',
            'Eng (2) Thrust Reverser',
        ), available)

    def derive(self,
               e1_dep_all=M('Eng (1) Thrust Reverser Deployed'),
               e1_dep_lft=M('Eng (1) Thrust Reverser (L) Deployed'),
               e1_dep_rgt=M('Eng (1) Thrust Reverser (R) Deployed'),
               e1_ulk_all=M('Eng (1) Thrust Reverser Unlocked'),
               e1_ulk_lft=M('Eng (1) Thrust Reverser (L) Unlocked'),
               e1_ulk_rgt=M('Eng (1) Thrust Reverser (R) Unlocked'),
               e1_tst_all=M('Eng (1) Thrust Reverser In Transit'),
               e2_dep_all=M('Eng (2) Thrust Reverser Deployed'),
               e2_dep_lft=M('Eng (2) Thrust Reverser (L) Deployed'),
               e2_dep_rgt=M('Eng (2) Thrust Reverser (R) Deployed'),
               e2_ulk_all=M('Eng (2) Thrust Reverser Unlocked'),
               e2_ulk_lft=M('Eng (2) Thrust Reverser (L) Unlocked'),
               e2_ulk_rgt=M('Eng (2) Thrust Reverser (R) Unlocked'),
               e2_tst_all=M('Eng (2) Thrust Reverser In Transit'),
               e3_dep_all=M('Eng (3) Thrust Reverser Deployed'),
               e3_dep_lft=M('Eng (3) Thrust Reverser (L) Deployed'),
               e3_dep_rgt=M('Eng (3) Thrust Reverser (R) Deployed'),
               e3_ulk_all=M('Eng (3) Thrust Reverser Unlocked'),
               e3_ulk_lft=M('Eng (3) Thrust Reverser (L) Unlocked'),
               e3_ulk_rgt=M('Eng (3) Thrust Reverser (R) Unlocked'),
               e3_tst_all=M('Eng (3) Thrust Reverser In Transit'),
               e4_dep_all=M('Eng (4) Thrust Reverser Deployed'),
               e4_dep_lft=M('Eng (4) Thrust Reverser (L) Deployed'),
               e4_dep_rgt=M('Eng (4) Thrust Reverser (R) Deployed'),
               e4_ulk_all=M('Eng (4) Thrust Reverser Unlocked'),
               e4_ulk_lft=M('Eng (4) Thrust Reverser (L) Unlocked'),
               e4_ulk_rgt=M('Eng (4) Thrust Reverser (R) Unlocked'),
               e4_tst_all=M('Eng (4) Thrust Reverser In Transit'),
               e1_status=M('Eng (1) Thrust Reverser'),
               e2_status=M('Eng (2) Thrust Reverser'),):

        deployed_params = (e1_dep_all, e1_dep_lft, e1_dep_rgt, e2_dep_all,
                           e2_dep_lft, e2_dep_rgt, e3_dep_all, e3_dep_lft,
                           e3_dep_rgt, e4_dep_all, e4_dep_lft, e4_dep_rgt,
                           e1_status, e2_status)

        deployed_stack = vstack_params_where_state(*[(d, 'Deployed') for d in deployed_params])

        unlocked_params = (e1_ulk_all, e1_ulk_lft, e1_ulk_rgt, e2_ulk_all,
                           e2_ulk_lft, e2_ulk_rgt, e3_ulk_all, e3_ulk_lft,
                           e3_ulk_rgt, e4_ulk_all, e4_ulk_lft, e4_ulk_rgt)

        array = np_ma_zeros_like(deployed_stack[0], dtype=np.short)
        stacks = [deployed_stack]

        if any(unlocked_params):
            unlocked_stack = vstack_params_where_state(*[(p, 'Unlocked') for p in unlocked_params])
            array = np.ma.where(unlocked_stack.any(axis=0), 1, array)
            stacks.append(unlocked_stack)

        array = np.ma.where(deployed_stack.any(axis=0), 1, array)
        array = np.ma.where(deployed_stack.all(axis=0), 2, array)

        # update with any transit params
        if any((e1_tst_all, e2_tst_all, e3_tst_all, e4_tst_all)):
            transit_stack = vstack_params_where_state(
                (e1_tst_all, 'In Transit'), (e2_tst_all, 'In Transit'),
                (e3_tst_all, 'In Transit'), (e4_tst_all, 'In Transit'),
                (e1_status, 'In Transit'), (e2_status, 'In Transit'),
            )
            array = np.ma.where(transit_stack.any(axis=0), 1, array)
            stacks.append(transit_stack)

        mask_stack = np.ma.concatenate(stacks, axis=0)

        # mask indexes with greater than 50% masked values
        mask = np.ma.where(mask_stack.mask.sum(axis=0).astype(float) / len(mask_stack) * 100 > 50, 1, 0)
        self.array = array
        self.array.mask = mask


class TAWSAlert(MultistateDerivedParameterNode):
    '''
    Merging all available TAWS alert signals into a single parameter for
    subsequent monitoring.
    '''
    name = 'TAWS Alert'
    values_mapping = {
        0: '-',
        1: 'Alert'}

    @classmethod
    def can_operate(cls, available):
        return any_of(['TAWS Caution Terrain',
                       'TAWS Caution',
                       'TAWS Dont Sink',
                       'TAWS Glideslope',
                       'TAWS Predictive Windshear',
                       'TAWS Pull Up',
                       'TAWS Sink Rate',
                       'TAWS Terrain',
                       'TAWS Terrain Warning Amber',
                       'TAWS Terrain Pull Up',
                       'TAWS Terrain Warning Red',
                       'TAWS Too Low Flap',
                       'TAWS Too Low Gear',
                       'TAWS Too Low Terrain',
                       'TAWS Windshear Warning',
                       ],
                      available)

    def derive(self, airs=S('Airborne'),
               taws_caution_terrain=M('TAWS Caution Terrain'),
               taws_caution=M('TAWS Caution'),
               taws_dont_sink=M('TAWS Dont Sink'),
               taws_glideslope=M('TAWS Glideslope'),
               taws_predictive_windshear=M('TAWS Predictive Windshear'),
               taws_pull_up=M('TAWS Pull Up'),
               taws_sink_rate=M('TAWS Sink Rate'),
               taws_terrain_pull_up=M('TAWS Terrain Pull Up'),
               taws_terrain_warning_amber=M('TAWS Terrain Warning Amber'),
               taws_terrain_warning_red=M('TAWS Terrain Warning Red'),
               taws_terrain=M('TAWS Terrain'),
               taws_too_low_flap=M('TAWS Too Low Flap'),
               taws_too_low_gear=M('TAWS Too Low Gear'),
               taws_too_low_terrain=M('TAWS Too Low Terrain'),
               taws_windshear_warning=M('TAWS Windshear Warning')):

        params_state = vstack_params_where_state(
            (taws_caution_terrain, 'Caution'),
            (taws_caution, 'Caution'),
            (taws_dont_sink, 'Warning'),
            (taws_glideslope, 'Warning'),
            (taws_predictive_windshear, 'Caution'),
            (taws_predictive_windshear, 'Warning'),
            (taws_pull_up, 'Warning'),
            (taws_sink_rate, 'Warning'),
            (taws_terrain_pull_up, 'Warning'),
            (taws_terrain_warning_amber, 'Warning'),
            (taws_terrain_warning_red, 'Warning'),
            (taws_terrain, 'Warning'),
            (taws_too_low_flap, 'Warning'),
            (taws_too_low_gear, 'Warning'),
            (taws_too_low_terrain, 'Warning'),
            (taws_windshear_warning, 'Warning'),
        )
        res = params_state.any(axis=0)

        self.array = np_ma_masked_zeros_like(params_state[0], dtype=np.short)
        if airs:
            for air in airs:
                self.array[air.slice] = res[air.slice]


class TAWSDontSink(MultistateDerivedParameterNode):
    name = 'TAWS Dont Sink'

    values_mapping = {
        0: '-',
        1: 'Warning',
    }

    @classmethod
    def can_operate(cls, available):
        return ('TAWS (L) Dont Sink' in available) or \
               ('TAWS (R) Dont Sink' in available)

    def derive(self, taws_l_dont_sink=M('TAWS (L) Dont Sink'),
               taws_r_dont_sink=M('TAWS (R) Dont Sink')):
        self.array = vstack_params_where_state(
            (taws_l_dont_sink, 'Warning'),
            (taws_r_dont_sink, 'Warning'),
        ).any(axis=0)


class TAWSGlideslopeCancel(MultistateDerivedParameterNode):
    name = 'TAWS Glideslope Cancel'

    values_mapping = {
        0: '-',
        1: 'Cancel',
    }

    @classmethod
    def can_operate(cls, available):
        return ('TAWS (L) Glideslope Cancel' in available) or \
               ('TAWS (R) Glideslope Cancel' in available)

    def derive(self, taws_l_gs=M('TAWS (L) Glideslope Cancel'),
               taws_r_gs=M('TAWS (R) Glideslope Cancel')):
        self.array = vstack_params_where_state(
            (taws_l_gs, 'Cancel'),
            (taws_r_gs, 'Cancel'),
        ).any(axis=0)


class TAWSTooLowGear(MultistateDerivedParameterNode):
    name = 'TAWS Too Low Gear'

    values_mapping = {
        0: '-',
        1: 'Warning',
    }

    @classmethod
    def can_operate(cls, available):
        return ('TAWS (L) Too Low Gear' in available) or \
               ('TAWS (R) Too Low Gear' in available)

    def derive(self, taws_l_gear=M('TAWS (L) Too Low Gear'),
               taws_r_gear=M('TAWS (R) Too Low Gear')):
        self.array = vstack_params_where_state(
            (taws_l_gear, 'Warning'),
            (taws_r_gear, 'Warning'),
        ).any(axis=0)


class TakeoffConfigurationWarning(MultistateDerivedParameterNode):
    '''
    Merging all available Takeoff Configuration Warning signals into a single
    parameter for subsequent monitoring.
    '''
    values_mapping = {
        0: '-',
        1: 'Warning',
    }

    @classmethod
    def can_operate(cls, available):
        return any_of(['Takeoff Configuration Stabilizer Warning',
                       'Takeoff Configuration Parking Brake Warning',
                       'Takeoff Configuration Flap Warning',
                       'Takeoff Configuration Gear Warning',
                       'Takeoff Configuration AP Warning',
                       'Takeoff Configuration Aileron Warning',
                       'Takeoff Configuration Rudder Warning',
                       'Takeoff Configuration Spoiler Warning'],
                      available)

    def derive(self, stabilizer=M('Takeoff Configuration Stabilizer Warning'),
               parking_brake=M('Takeoff Configuration Parking Brake Warning'),
               flap=M('Takeoff Configuration Flap Warning'),
               gear=M('Takeoff Configuration Gear Warning'),
               ap=M('Takeoff Configuration AP Warning'),
               ail=M('Takeoff Configuration Aileron Warning'),
               rudder=M('Takeoff Configuration Rudder Warning'),
               spoiler=M('Takeoff Configuration Rudder Warning')):
        params_state = vstack_params_where_state(
            (stabilizer, 'Warning'),
            (parking_brake, 'Warning'),
            (flap, 'Warning'),
            (gear, 'Warning'),
            (ap, 'Warning'),
            (ail, 'Warning'),
            (rudder, 'Warning'),
            (spoiler, 'Warning'))
        self.array = params_state.any(axis=0)


class TCASFailure(MultistateDerivedParameterNode):
    name = 'TCAS Failure'

    values_mapping = {
        0: '-',
        1: 'Failed',
    }

    @classmethod
    def can_operate(cls, available):
        return ('TCAS (L) Failure' in available) or \
               ('TCAS (R) Failure' in available)

    def derive(self, tcas_l_failure=M('TCAS (L) Failure'),
               tcas_r_failure=M('TCAS (R) Failure')):
        self.array = vstack_params_where_state(
            (tcas_l_failure, 'Failed'),
            (tcas_r_failure, 'Failed'),
        ).any(axis=0)


class SpeedControl(MultistateDerivedParameterNode):

    values_mapping = {0: 'Manual', 1: 'Auto'}

    @classmethod
    def can_operate(cls, available):

        return any_of((
            'Speed Control Auto',
            'Speed Control Manual',
            'Speed Control (1) Auto',
            'Speed Control (1) Manual',
            'Speed Control (2) Auto',
            'Speed Control (2) Manual',
        ), available)

    def derive(self,
               sc0a=M('Speed Control Auto'),
               sc0m=M('Speed Control Manual'),
               sc1a=M('Speed Control (1) Auto'),
               sc1m=M('Speed Control (1) Manual'),
               sc2a=M('Speed Control (2) Auto'),
               sc2m=M('Speed Control (2) Manual')):

        self.array = vstack_params_where_state(
            (sc0a, 'Auto'), (sc0m, 'Auto'),
            (sc1a, 'Auto'), (sc1m, 'Auto'),
            (sc2a, 'Auto'), (sc2m, 'Auto'),
        ).any(axis=0).astype(np.int)
