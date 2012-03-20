import logging
import numpy as np

from analysis_engine import settings
from analysis_engine.settings import CONTROL_FORCE_THRESHOLD, RATE_OF_TURN_FOR_TAXI_TURNS

from analysis_engine.node import KeyPointValueNode, KPV, KTI, P, S
from analysis_engine.library import (clip, coreg, 
                                     index_at_value, max_abs_value,
                                     max_continuous_unmasked, max_value,
                                     min_value, repair_mask, 
                                     slice_samples, 
                                     slices_above,
                                     slices_below,
                                     slices_from_to,
                                     subslice,
                                     value_at_index)

class AccelerationLateralTaxiingMax(KeyPointValueNode):
    def derive(self, acc_lat = P('Acceleration Lateral'), taxi = S('Taxi')):
        self.create_kpvs_within_slices(acc_lat.array, taxi, max_value)
    
        
class AccelerationNormal20FtToGroundMax(KeyPointValueNode):
    name = 'Acceleration Normal 20 Ft To Ground Max' # not required?
    def derive(self, acceleration_normal=P('Acceleration Normal'),
               alt_aal=P('Altitude AAL')):
        # Q: Is from 20 Ft to 0 Ft of Alt AAL the same as '20 Ft To Ground'?
        self.create_kpvs_within_slices(acceleration_normal.array,
                                       alt_aal.slices_from_to(20, 0),
                                       max_value)


class AccelerationNormalAtTouchdown(KeyPointValueNode):
    def derive(self, acc=P('Acceleration Normal'), tdwns=KTI('Touchdown')):
        for tdwn in tdwns:
            dt=2.0 # Half width of range to scan across for peak acceleration.
            from_index = max(int(tdwn.index-dt*acc.hz), 0)
            to_index = min(int(tdwn.index+dt*acc.hz)+1, len(acc.array))
            bump_accel = acc.array[from_index:to_index]
            bump_index = np.ma.argmax(bump_accel)
            peak = bump_accel[bump_index]
            self.create_kpv(from_index + bump_index, peak)


class Airspeed1000To500FtMax(KeyPointValueNode):
    def derive(self, speed=P('Airspeed'), alt_aal=P('Altitude AAL')):
        self.create_kpvs_within_slices(speed.array,
                                       alt_aal.slices_from_to(1000, 500),
                                       max_value)


class Airspeed2000To30FtMin(KeyPointValueNode):
    def derive(self, speed=P('Airspeed'), alt_aal=P('Altitude AAL')):
            self.create_kpvs_within_slices(speed.array,
                                           alt_aal.slices_from_to(2000, 30),
                                           min_value)


class Airspeed35To50FtMax(KeyPointValueNode):
    def derive(self, speed=P('Airspeed'), alt_aal=P('Altitude AAL')):
            self.create_kpvs_within_slices(speed.array,
                                           alt_aal.slices_from_to(500, 50),
                                           max_value) 


class Airspeed50To1000FtMax(KeyPointValueNode):
    def derive(self, speed=P('Airspeed'), alt_aal=P('Altitude AAL')):
            self.create_kpvs_within_slices(speed.array,
                                           alt_aal.slices_from_to(50, 1000),
                                           max_value) 

class Airspeed400To1500FtMin(KeyPointValueNode):
    def derive(self, speed=P('Airspeed'), alt_aal=P('Altitude AAL')):
            self.create_kpvs_within_slices(speed.array,
                                           alt_aal.slices_from_to(400, 1500),
                                           min_value)


class Airspeed400To50FtMax(KeyPointValueNode):
    def derive(self, speed=P('Airspeed'), alt_aal=P('Altitude AAL')):
            self.create_kpvs_within_slices(speed.array,
                                           alt_aal.slices_from_to(500, 50),
                                           max_value)


class Airspeed500To50FtMax(KeyPointValueNode):
    def derive(self, speed=P('Airspeed'), alt_aal=P('Altitude AAL')):
            self.create_kpvs_within_slices(speed.array,
                                           alt_aal.slices_from_to(500, 50),
                                           max_value) 


#-------------------------------------------------------------------------------
# Set of gear speed KPVs, which should better reflect what we need to monitor.
#TODO: Review these
#-------------------------------------------------------------------------------
class AirspeedAsGearRetracting(KeyPointValueNode):
    def derive(self, airspeed=P('Airspeed'), gear_ret=KTI('Gear Retracting')):
        self.create_kpvs_at_ktis(airspeed.array, gear_ret)


class AirspeedAsGearSelectedUp(KeyPointValueNode):
    # Q: Does this mean a KPV will be created on switching the discrete from 0
    # to 1?
    def derive(self, airspeed=P('Airspeed'), 
               gear_sel_up=P('Gear Selected Up')):
        self.create_kpvs_at_ktis(airspeed.array, gear_sel_up)


class AirspeedWithGearDownMax(KeyPointValueNode):
    def derive(self, airspeed=P('Airspeed'), gear=P('Gear Down'), 
               airs=S('Airborne')):
        for air in airs:
            downs = np.ma.clump_unmasked(np.ma.masked_equal(gear.array[air.slice], 0.0))
            for down in downs:
                index = np.ma.argmax(airspeed.array[air.slice][down])
                value = airspeed.array[air.slice][down][index]
                self.create_kpv(int(air.slice.start)+down.start+index, value)


class AirspeedWithGearSelectedDownMax(KeyPointValueNode):
    def derive(self, airspeed=P('Airspeed'),
               gear_down=P('Gear Down')):
        '''
        First programmed for 737-5 frame, where gear selection is not
        available, hence based on gear position. 
        Expects 'Gear Down' to be a discrete parameter.
        '''
        # Mask values where gear_down != 1.
        airspeed.array[gear_down.array != 1] = np.ma.masked
        index, value = max_value(airspeed.array)
        self.create_kpv(index, value)

        
class AirspeedAsGearExtending(KeyPointValueNode):
    def derive(self, airspeed=P('Airspeed'), gear_ext=KTI('Gear Extending')):
        self.create_kpvs_at_ktis(airspeed.array, gear_ext)


class AirspeedAsGearSelectedDown(KeyPointValueNode):
    # Q: Does this mean a KPV will be created on switching the discrete from 0
    # to 1?
    def derive(self, airspeed=P('Airspeed'),
               gear_sel_down=KTI('Gear Selected Down')):
        self.create_kpvs_at_ktis(airspeed.array, gear_sel_down)

#-------------------------------------------------------------------------------

class AirspeedAtLiftoff(KeyPointValueNode):
    '''
    DJ suggested TailWindAtLiftoff would complement this parameter when used
    for 'Speed high at liftoff' events.
    '''
    def derive(self, airspeed=P('Airspeed'), liftoffs=KTI('Liftoff')):
        self.create_kpvs_at_ktis(airspeed.array, liftoffs)


class AirspeedAtTouchdown(KeyPointValueNode):
    def derive(self, airspeed=P('Airspeed'), touchdowns=KTI('Touchdown')):
        self.create_kpvs_at_ktis(airspeed.array, touchdowns)


class AirspeedMax(KeyPointValueNode):
    def derive(self, speed=P('Airspeed'), airs=S('Airborne')):
        # For commented version, see GlideslopeDeviation1500To1000FtMax
        self.create_kpvs_within_slices(speed.array, airs, max_value)


class AirspeedMax3Sec(KeyPointValueNode):
    def derive(self, speed=P('Airspeed'), airs=S('Airborne')):
        self.create_kpvs_within_slices(clip(speed.array, 3.0, speed.hz), airs, max_value)


class AltitudeAtTouchdown(KeyPointValueNode):
    def derive(self, alt_std=P('Altitude STD'), touchdowns=KTI('Touchdown')):
        self.create_kpvs_at_ktis(alt_std.array, touchdowns)


class AutopilotEngaged1AtLiftoff(KeyPointValueNode):
    name = 'Autopilot Engaged 1 At Liftoff'
    def derive(self, autopilot=KTI('Autopilot Engaged 1'),
               liftoffs=P('Liftoff')):
        self.create_kpvs_at_ktis(autopilot.array, liftoffs)


class AutopilotEngaged2AtLiftoff(KeyPointValueNode):
    name = 'Autopilot Engaged 2 At Liftoff'
    def derive(self, autopilot=KTI('Autopilot Engaged 2'),
               liftoffs=P('Liftoff')):
        self.create_kpvs_at_ktis(autopilot.array, liftoffs)


class AutopilotEngaged1AtTouchdown(KeyPointValueNode):
    name = 'Autopilot Engaged 1 At Touchdown'
    def derive(self, autopilot=KTI('Autopilot Engaged 1'),
               touchdowns=P('Touchdown')):
        self.create_kpvs_at_ktis(autopilot.array, touchdowns)


class AutopilotEngaged2AtTouchdown(KeyPointValueNode):
    name = 'Autopilot Engaged 2 At Touchdown'
    def derive(self, autopilot=KTI('Autopilot Engaged 2'),
               touchdowns=P('Touchdown')):
        self.create_kpvs_at_ktis(autopilot.array, touchdowns)


class ControlColumnStiffness(KeyPointValueNode):
    """
    The control force and displacement of the flying controls should follow a
    predictable relationship. This parameter is included to identify
    stiffness in the controls in flight.
    """
    def derive(self,
               force=P('Control Column Force'),
               disp=P('Control Column'),
               fast=S('Fast')):

        #-------------------------------------------------------------------
        # TEST OUTPUT TO CSV FILE FOR DEBUGGING ONLY
        # TODO: REMOVE THIS SECTION BEFORE RELEASE
        #-------------------------------------------------------------------
        import csv
        spam = csv.writer(open('cheese.csv', 'wb'))
        spam.writerow(['ref','column', 'force'])
        #-------------------------------------------------------------------
        # TEST OUTPUT TO CSV FILE FOR DEBUGGING ONLY
        # TODO: REMOVE THIS SECTION BEFORE RELEASE
        #-------------------------------------------------------------------
        

        # We only test during high speed operation to avoid "testing" the
        # full and free movements before flight.
        for speedy in fast:
            # We look for forces above a threshold to detect manual input.
            # This is better than looking for movement, as in the case of
            # stiff controls there is more force but less movement, hence
            # using a movement threshold will tend to be suppressed in the
            # cases we are looking to detect.
            push = force.array[speedy.slice]
            column = disp.array[speedy.slice]
            
            moves = np.ma.clump_unmasked(
                np.ma.masked_less(np.ma.abs(push),
                                  CONTROL_FORCE_THRESHOLD))
            for move in moves:
                if slice_samples(move) < 10:
                    continue
                corr, slope, off = \
                    coreg(push[move], indep_var=column[move], force_zero=True)
                if corr>0.85:  # This checks the data looks sound.
                    when = np.ma.argmax(np.ma.abs(push[move]))
                    self.create_kpv(speedy.slice.start+move.start+when, slope)
                    
                    #-------------------------------------------------------------------
                    # TEST OUTPUT TO CSV FILE FOR DEBUGGING ONLY
                    # TODO: REMOVE THIS SECTION BEFORE RELEASE
                    #-------------------------------------------------------------------
                    n = speedy.slice.start+move.start
                    for showme in range(0, slice_samples(move)):
                        spam.writerow([n+showme,column[move][showme],push[move][showme]])
                     #-------------------------------------------------------------------
                    # TEST OUTPUT TO CSV FILE FOR DEBUGGING ONLY
                    # TODO: REMOVE THIS SECTION BEFORE RELEASE
                    #-------------------------------------------------------------------
                    
                    

class GoAroundAltitude(KeyPointValueNode):
    def derive(self, gas=KTI('Go Around'),
                alt_std=P('Altitude AAL'),
                alt_rad=P('Altitude Radio')):
        for ga in gas:
            if alt_rad:
                index = np.ma.argmin(alt_rad.array[ga.index])
                pit = alt_rad.array[ga.index+index]
            else:
                index = np.ma.argmin(alt_std.array[ga.index])
                pit = alt_std.array[ga.index+index]
            self.create_kpv(index, pit)
         

class HeadingAtLanding(KeyPointValueNode):
    """
    We take the median heading, as this avoids problems with drift just
    before touchdown and turning off the runway. The value is "assigned" to a
    time midway through the landing phase.
    """
    def derive(self, lands=S('Landing'), 
               head=P('Heading Continuous')):
        for land in lands:
            land_head = np.ma.median(head.array[land.slice])
            land_index = (land.slice.start + land.slice.stop)/2.0
            self.create_kpv(land_index, land_head%360.0)


class HeadingAtLowestPointOnApproach(KeyPointValueNode):
    """
    The approach phase has been found already. Here we take the heading at
    the lowest point reached in the approach. This may not be a go-around, if
    the aircraft did not climb 500ft before the next approach to landing.
    """
    def derive(self, head=P('Heading Continuous'), 
               go_arounds=KTI('Go Around')):
        self.create_kpvs_at_ktis(head.array, go_arounds)
    """
    def derive(self, head=P('Heading Continuous'),
               lands=KTI('Approach And Landing Lowest Point')):
        self.create_kpvs_at_ktis(head.array, lands)
        """

class HeadingAtTakeoff(KeyPointValueNode):
    """
    We take the median heading, as this avoids problems with drift just
    after liftoff and turning onto the runway. The value is "assigned" to a
    time midway through the landing phase.
    """
    def derive(self, toffs=S('Takeoff'),
               head=P('Heading Continuous')):
        for toff in toffs:
            toff_head = np.ma.median(head.array[toff.slice])
            toff_index = (toff.slice.start + toff.slice.stop)/2.0
            self.create_kpv(toff_index, toff_head%360.0)


class LatitudeAtLanding(KeyPointValueNode):
    # Cannot use smoothed position as this causes circular dependancy.
    def derive(self, lat=P('Latitude'), lands=KTI('Landing Peak Deceleration')):
        '''
        While storing this is redundant due to geo-locating KeyPointValues, it is
        used in multiple Nodes to simplify their implementation.
        '''    
        self.create_kpvs_at_ktis(lat.array, lands)
            

class LongitudeAtLanding(KeyPointValueNode):
    # Cannot use smoothed position as this causes circular dependancy.
    def derive(self, lon=P('Longitude'),lands=KTI('Landing Peak Deceleration')):
        '''
        While storing this is redundant due to geo-locating KeyPointValues, 
        it is used in multiple Nodes to simplify their implementation.
        '''       
        self.create_kpvs_at_ktis(lon.array, lands)


class LatitudeAtLiftoff(KeyPointValueNode):
    def derive(self, lat=P('Latitude'),
               liftoffs=KTI('Liftoff')):
        self.create_kpvs_at_ktis(lat.array, liftoffs)


class LongitudeAtLiftoff(KeyPointValueNode):
    '''
    While storing this is redundant due to geo-locating KeyPointValues, it is
    used in multiple Nodes to simplify their implementation.
    '''    
    def derive(self, lon=P('Longitude'),
               liftoffs=KTI('Liftoff')):
        self.create_kpvs_at_ktis(lon.array, liftoffs)


class ILSFrequencyOnApproach(KeyPointValueNode):
    """
    The period when the aircraft was continuously established on the ILS and
    descending to the minimum point on the approach is already defined as a
    flight phase. This KPV just picks up the frequency tuned at that point.
    """
    name='ILS Frequency On Approach' #  Set here to ensure "ILS" in uppercase.
    def derive(self, establishes=S('ILS Localizer Established'),
              f1=P('ILS (L) Frequency'),f2=P('ILS (R) Frequency'),
              lowest=KTI('Approach And Landing Lowest Point'),
              ils_frq=P('ILS Frequency')):
        
        for established in establishes:
            # For the final period of operation of the ILS during this
            # approach, the ILS frequency was:
            freq=np.ma.median(ils_frq.array[established.slice])
            # Note median picks the value most commonly recorded, so allows
            # for some masked values and perhaps one or two rogue values.

            # Identify the KPV as relating to the start of this ILS approach
            self.create_kpv(established.slice.start, freq)


class LatitudeAtLowestPointOnApproach(KeyPointValueNode):
    # Cannot use smoothed position as this causes circular dependancy.
    def derive(self, lat=P('Latitude'), 
               go_arounds=KTI('Go Around')):
        self.create_kpvs_at_ktis(lat.array, go_arounds)
    """
    def derive(self, lat=P('Latitude'), 
               lands=KTI('Approach And Landing Lowest Point')):
        self.create_kpvs_at_ktis(lat.array, lands)
        """    

class LongitudeAtLowestPointOnApproach(KeyPointValueNode):
    # Cannot use smoothed position as this causes circular dependancy.
    def derive(self, lon=P('Longitude'), 
               go_arounds=KTI('Go Around')):
        self.create_kpvs_at_ktis(lon.array, go_arounds)
    """
    def derive(self, lon=P('Longitude'), 
               lands=KTI('Approach And Landing Lowest Point')):
        self.create_kpvs_at_ktis(lon.array, lands)
        """
   
class FlapAtLiftoff(KeyPointValueNode):
    def derive(self, flap=P('Flap'), liftoffs=KTI('Liftoff')):
        self.create_kpvs_at_ktis(flap.array, liftoffs)
    

class AccelerationNormalFtTo35FtMax(KeyPointValueNode): # Q: Name?
    def derive(self, norm_g=P('Acceleration Normal'), alt_rad=P('Altitude Radio')):
        return NotImplemented


class AccelerationNormalAirborneMax(KeyPointValueNode):
    def derive(self, accel=P('Acceleration Normal'), airborne=S('Airborne')):
        self.create_kpvs_within_slices(accel.array, airborne, max_value)


class AccelerationLongitudinalPeakTakeoff(KeyPointValueNode):
    def derive(self, takeoff=S('Takeoff'),
               accel=P('Acceleration Longitudinal')):
        self.create_kpvs_within_slices(accel.array, takeoff, max_value)


class DecelerationLongitudinalPeakLanding(KeyPointValueNode):
    def derive(self, landing=S('Landing'),
               accel=P('Acceleration Longitudinal')):
        self.create_kpvs_within_slices(accel.array, landing, min_value)


class AccelerationNormalAirborneMin(KeyPointValueNode):
    def derive(self, accel=P('Acceleration Normal'), airborne=S('Airborne')):
        self.create_kpvs_within_slices(accel.array, airborne, min_value)


'''
class Pitch35To400FtMax(KeyPointValueNode):
    def derive(self, pitch=P('Pitch'), alt_rad=P('Altitude Radio')):
        return NotImplemented
'''


class AltitudeWithFlapsMax(KeyPointValueNode):
    #TODO: TESTS
    """
    FIXME: It's max Altitude not Max Flaps
    """
    def derive(self, flap=P('Flap'), alt_std=P('Altitude STD')):
        '''
        flap_set = np.ma.masked_less_equal(flap.array, 0)
        # Failed on the line below with the following error...
        # TypeError: ufunc 'bitwise_or' not supported for the input types, and the inputs could not be safely coerced to any supported types according to the casting rule 'safe'
        index, value = max_value(alt_std.array | flap_set)
        #...hence the noddy rewrite.
        self.create_kpv(index, value)
        '''
        alt_flap = alt_std.array * np.ma.minimum(flap.array,1.0)
        index = np.ma.argmax(alt_flap)
        value = alt_flap[index]
        self.create_kpv(index, value)
        
        
class MachMax(KeyPointValueNode):
    name = 'Mach Max'
    def derive(self, mach=P('Mach'), airs=S('Airborne')):
        self.create_kpvs_within_slices(mach.array, airs, max_value)


class MachMax3Sec(KeyPointValueNode):
    def derive(self, mach=P('Mach'), airs=S('Airborne')):
        self.create_kpvs_within_slices(clip(mach.array, 3.0, mach.hz),
                                       airs, max_value)



class AltitudeAtMachMax(KeyPointValueNode):
    name = 'Altitude At Mach Max'
    def derive(self, alt_std=P('Altitude STD'), max_mach=KPV('Mach Max')):
        # Aligns Altitude to Mach to ensure we have the most accurate
        # altitude reading at the point of Maximum Mach
        self.create_kpvs_at_kpvs(alt_std.array, max_mach)


'''
# See taxi speeds - redundant?
class GroundSpeedOnGroundMax(KeyPointValueNode):
    def derive(self, groundspeed=P('Groundspeed'), on_grounds=S('On Ground')):
        self.create_kpvs_within_slices(groundspeed.array, on_grounds, max_value)
'''

class FlapAtTouchdown(KeyPointValueNode):
    def derive(self, flap=P('Flap'), touchdowns=KTI('Touchdown')):
        self.create_kpvs_at_ktis(flap.array, touchdowns)


#################
# TODO: Review whether Engine measurements should be overall or for sections 
# in flight (e.g. split by airborne / on ground?)

class EngEGTMax(KeyPointValueNode):
    name = 'Eng EGT Max'
    def derive(self, eng=P('Eng (*) EGT Max')):
        index, value = max_value(eng.array)
        self.create_kpv(index, value)


class EngN10FtToFL100Max(KeyPointValueNode):
    '''TODO: Test.'''
    name = 'Eng N1 0 Ft To FL100 Max'
    def derive(self, eng=P('Eng (*) N1 Max'), alt_std=P('Altitude STD')):
        self.create_kpvs_within_slices(eng.array,
                                       alt_std.slices_below(10000),
                                       max_value)


class EngEPR0FtToFL100Max(KeyPointValueNode):
    '''TODO: Test.'''
    name = 'Eng EPR 0 Ft To FL100 Max'
    def derive(self, eng=P('Eng (*) EPR Max'), alt_std=P('Altitude STD')):
        self.create_kpvs_within_slices(eng.array,
                                       alt_std.slices_from_to(0, 10000),
                                       max_value)


class EngTorque0FtToFL100Max(KeyPointValueNode):
    '''TODO: Test.'''
    name = 'Eng Torque 0 Ft To FL100 Max'
    def derive(self, eng=P('Eng (*) Torque Max'), alt_std=P('Altitude STD')):
        self.create_kpvs_within_slices(eng.array,
                                       alt_std.slices_from_to(0, 10000),
                                       max_value)


class EngEPR500FtToTouchdownMin(KeyPointValueNode):
    name = 'Eng EPR 500 Ft To Touchdown Min'
    def derive(self, eng=P('Eng (*) EPR Min'), alt_aal=P('Altitude AAL')):
        self.create_kpvs_within_slices(eng.array,
                                       alt_aal.slices_from_to(500, 0),
                                       min_value)

class EngN1500FtToTouchdownMin(KeyPointValueNode):
    name = 'Eng N1 500 Ft To Touchdown Min'
    def derive(self, eng=P('Eng (*) N1 Min'), alt_aal=P('Altitude AAL')):
        self.create_kpvs_within_slices(eng.array,
                                       alt_aal.slices_from_to(500, 0),
                                       min_value)


class EngTorque500FtToTouchdownMax(KeyPointValueNode):
    def derive(self, eng=P('Eng (*) Torque Max'), alt_aal=P('Altitude AAL')):
        self.create_kpvs_within_slices(eng.array,
                                       alt_aal.slices_from_to(500, 0),
                                       min_value)


class EngTorque500FtToTouchdownMin(KeyPointValueNode):
    def derive(self, eng=P('Eng (*) Torque Min'), alt_aal=P('Altitude AAL')):
        self.create_kpvs_within_slices(eng.array,
                                       alt_aal.slices_from_to(500, 0),
                                       min_value)
        

class Eng_N1MaxDurationUnder60PercentAfterTouchdown(KeyPointValueNode): ##was named: EngN1CooldownDuration
    """
    Max duration N1 below 60% after Touchdown for engine cooldown. Using 60%
    allows for cooldown after use of Reverse Thrust.
    
    Evaluated for each engine to account for single engine taxi-in.
    
    Note: Assumes that all Engines are recorded at the same frequency.
    
    TODO: Eng_Stop KTI
    TODO: Similar KPV for duration between engine under 60 percent and engine shutdown
    """
    NAME_FORMAT = "Eng (%(eng_num)d) N1 Max Duration Under 60 Percent After Touchdown"
    NAME_VALUES = {'eng_num': [1,2,3,4]}
    
    @classmethod
    def can_operate(cls, available):
        return 'Touchdown' in available and \
               'Eng (*) Stop' in available and \
               ('Eng (1) N1' in available or \
                'Eng (2) N1' in available or \
                'Eng (3) N1' in available or \
                'Eng (4) N1' in available)
    
    def derive(self, 
               eng1=P('Eng (1) N1'),
               eng2=P('Eng (2) N1'),
               eng3=P('Eng (3) N1'),
               eng4=P('Eng (4) N1'),
               tdwn=KTI('Touchdown'), engines_stop=KTI('Eng (*) Stop')):
                
        for eng_num, eng in enumerate((eng1,eng2,eng3,eng4), start=1):
            if eng is None:
                continue # engine not available on this aircraft
            eng_stop = engines_stop.get(name='Eng (%d) Stop' % eng_num)
            if not eng_stop:
                #Q: Should we measure until the end of the flight anyway? (probably not)
                logging.debug("Engine %d did not stop on this flight, cannot measure KPV", eng_num)
                continue
            
            eng_array = repair_mask(eng.array)
            eng_below_60 = np.ma.masked_greater(eng_array, 60)
            # measure duration between final touchdown and engine stop
            touchdown_to_stop_slice = max_continuous_unmasked(
                eng_below_60, slice(tdwn[-1].index, eng_stop[0].index))
            if touchdown_to_stop_slice:
                #TODO future storage of slice: self.slice = touchdown_to_stop_slice
                touchdown_to_stop_duration = (touchdown_to_stop_slice.stop - \
                                        touchdown_to_stop_slice.start) / self.hz
                self.create_kpv(touchdown_to_stop_slice.start,
                                touchdown_to_stop_duration, eng_num=eng_num)
            else:
                # create KPV of 0 seconds
                self.create_kpv(eng_stop[0].index, 0.0, eng_num=eng_num)
        
        
class EngN1Max(KeyPointValueNode):
    name = 'Eng N1 Max'
    def derive(self, eng=P('Eng (*) N1 Max')):
        index, value = max_value(eng.array)
        self.create_kpv(index, value)

class EngN2Max(KeyPointValueNode):
    name = 'Eng N2 Max'
    def derive(self, eng=P('Eng (*) N2 Max')):
        index, value = max_value(eng.array)
        self.create_kpv(index, value)

class EngOilTempMax(KeyPointValueNode):
    name = 'Eng Oil Temp Max'
    def derive(self, eng=P('Eng (*) Oil Temp Max')):
        index, value = max_value(eng.array)
        self.create_kpv(index, value)


class EngVibN1Max(KeyPointValueNode):
    name = 'Eng Vib N1 Max'
    ##def derive(self, eng=P('Eng (*) Vib N1 Max'), fast=S('Fast')):
        ##for sect in fast:
            ##index, value = max_value(eng.array, sect.slice)
            ##self.create_kpv(index, value)
            
    def derive(self, eng=P('Eng (*) Vib N1 Max')):
        index, value = max_value(eng.array)
        self.create_kpv(index, value)
            
            
class EngVibN2Max(KeyPointValueNode):
    name = 'Eng Vib N2 Max'
    ##def derive(self, eng=P('Eng (*) Vib N2 Max'), fast=S('Fast')):
        ##for sect in fast:
            ##index, value = max_value(eng.array, sect.slice)
            ##self.create_kpv(index, value)
            
    def derive(self, eng=P('Eng (*) Vib N2 Max')):
        index, value = max_value(eng.array)
        self.create_kpv(index, value)
            
            
class FuelQtyAirborneMin(KeyPointValueNode):
    def derive(self, fuel=P('Fuel Qty'), airborne=S('Airborne')):
        self.create_kpvs_within_slices(fuel.array, airborne, min_value)

'''
See Heading at liftoff and touchdown - TODO: Remove if possible. DJ
class MagneticHeadingAtLiftOff(KeyPointValue):
    """ Shouldn't this be difference between aircraft heading and runway heading???
    """
    def derive(self, heading=P('Magnetic Heading'), liftoff=KTI('Liftoff')):
        return NotImplemented


class MagneticHeadingAtTouchdown(KeyPointValue):
    """ Shouldn't this be difference between aircraft heading and runway heading???
    """
    def derive(self, heading=P('Magnetic Heading'), touchdown=KTI('Touchdown')):
        return NotImplemented
'''
    
# TODO: Trouble with naming these
#class LatgOnGround??
#class Pitch rate (from 2 degrees of pitch?) to 35 feet minimum

    
'''Shortcut: copy n paste below!
(KeyPointValueNode):
    dependencies = []
    def derive(self, params):
        return NotImplemented
'''

##########################################
# KPV from DJ's Code

class AccelerationNormalMax(KeyPointValueNode):
    def derive(self, acc_norm=P('Acceleration Normal')):
        index, value = max_value(acc_norm.array)
        self.create_kpv(index, value)


class AccelerationNormalMax(KeyPointValueNode):
    def derive(self, acc_norm=P('Acceleration Normal')):
        index, value = max_value(acc_norm.array)
        self.create_kpv(index, value)


class RateOfClimbMax(KeyPointValueNode):
    '''
    .. TODO:: testcases
    '''
    def derive(self, rate_of_climb=P('Rate Of Climb'),
               climbing=S('Climbing')):
        #TODO: Merge with below RateOfDescentMax accepting a flightphase arg
        for climb in climbing:
            duration = climb.slice.stop - climb.slice.start
            if duration > settings.CLIMB_OR_DESCENT_MIN_DURATION:
                index, value = max_value(rate_of_climb.array, climb.slice)
                self.create_kpv(index, value)


class RateOfDescentMax(KeyPointValueNode):
    '''
    .. TODO:: testcases
    '''
    def derive(self, rate_of_climb=P('Rate Of Climb'),
               descending=S('Descending')):
        for descent in descending:
            duration = descent.slice.stop - descent.slice.start
            if duration > settings.CLIMB_OR_DESCENT_MIN_DURATION:
                index, value = min_value(rate_of_climb.array, descent.slice)
                self.create_kpv(index, value)

                
class AirspeedLevelFlightMax(KeyPointValueNode):
    def derive(self, airspeed=P('Airspeed'), level_flight=S('Level Flight')):
        for sect in level_flight:
            #TODO: Move LEVEL_FLIGHT_MIN_DURATION to LevelFlight
            #FlightPhaseNode so that only stable level flights are reported.
            duration = (sect.slice.stop - sect.slice.start)/self.frequency
            if duration > settings.LEVEL_FLIGHT_MIN_DURATION:
                # stable level flight
                index, value = max_value(airspeed.array, sect.slice)
                self.create_kpv(index, value)
            else:
                logging.debug("Level flight duration too short to create KPV")


class AccelerationNormalDuringTakeoffMax(KeyPointValueNode):
    def derive(self, acceleration_normal=P('Acceleration Normal'),
               takeoffs=S('Takeoff')):
        self.create_kpvs_within_slices(acceleration_normal.array, takeoffs,
                                       max_value)

        
class AltitudeMax(KeyPointValueNode):
    def derive(self, alt_std=P('Altitude STD'), airs=S('Airborne')):
        self.create_kpvs_within_slices(alt_std.array, airs, max_value)


# FIXME: Bounced Landing name duplicated between KPV and Section!
class BouncedLanding(KeyPointValueNode):
    def derive(self, bounced_landing=S('Bounced Landing Section')):
        return NotImplemented


class RateOfDescent500FtToTouchdownMax(KeyPointValueNode):
    def derive(self, roc=P('Rate Of Climb'), alt_aal=P('Altitude AAL')):
        self.create_kpvs_within_slices(roc.array, 
                                       alt_aal.slices_from_to(500, 0),
                                       min_value)


class RateOfDescent1000To500FtMax(KeyPointValueNode):
    def derive(self, roc=P('Rate Of Climb'),alt_aal=P('Altitude AAL')):
        self.create_kpvs_within_slices(roc.array,
                                       alt_aal.slices_from_to(1000, 500),
                                       min_value)


class RateOfDescent1000To50FtMax(KeyPointValueNode):
    def derive(self, roc=P('Rate Of Climb'),alt_aal=P('Altitude AAL')):
        self.create_kpvs_within_slices(roc.array,
                                       alt_aal.slices_from_to(1000, 50),
                                       min_value)


class RateOfDescent2000To1000FtMax(KeyPointValueNode):
    def derive(self, roc=P('Rate Of Climb'), alt_aal=P('Altitude AAL')):
        self.create_kpvs_within_slices(roc.array,
                                       alt_aal.slices_from_to(2000, 1000),
                                       min_value)


class DontSinkWarning(KeyPointValueNode):
    def derive(self, taws_dont_sink=P("TAWS Don't Sink")):
        return NotImplemented


class HeightAtConfigChangeLiftoffTo3500FtMin(KeyPointValueNode):
    #TODO: TESTS
    
    # This is height at FIRST config change - KPV is MIN(Height) at config change
    def derive(self, flap=P('Flap'), alt_aal=P('Altitude AAL'),
               airs=S('Airborne')):
        # use .diff and declare the heights at those indexes
        for air in airs: # Was "alt_aal.slices_from_to(0, 3500):" but triggered for flap changes on the ground.
            # find where flap changes
            change_indexes = np.ma.where(np.ma.diff(flap.array[air.slice]))[0]
            if len(change_indexes):
                # create at first change
                index = air.slice.start + change_indexes[0]
                self.create_kpv(air.slice.start + change_indexes[0], 
                                value_at_index(alt_aal.array, index))


class EngEGTTakeoffMax(KeyPointValueNode):
    name = 'Eng EGT Takeoff Max'
    def derive(self, eng_egt=P('Eng (*) EGT Max'), takeoffs=KTI('Takeoff')):
        self.create_kpvs_within_slices(eng_egt.array, takeoffs, max_value)


class TAWSGlideslopeWarning(KeyPointValueNode):
    name = 'TAWS Glideslope Warning'
    def derive(self, taws_glideslope=P('TAWS Glideslope')):
        return NotImplemented


class ILSGlideslopeDeviation1500To1000FtMax(KeyPointValueNode):
    name = 'ILS Glideslope Deviation 1500 To 1000 Ft Max'
    def derive(self, ils_glideslope=P('ILS Glideslope'),
               alt_aal = P('Altitude AAL'),
               estabs=S('ILS Glideslope Established')):
        # Find where the maximum (absolute) deviation occured and
        # store the actual value. We can do abs on the statistics to
        # normalise this, but retaining the sign will make it
        # possible to look for direction of errors at specific
        # airports.
        for estab in estabs:
            for band in slices_from_to(alt_aal.array[estab.slice],1500, 1000)[1]:
                kpv_slice=[slice(estab.slice.start+band.start, estab.slice.start+band.stop)]
                self.create_kpvs_within_slices(ils_glideslope.array,kpv_slice,max_abs_value)  


class ILSGlideslopeDeviationAbove1000FtMax(KeyPointValueNode):
    name = 'ILS Glideslope Deviation Above 1000 Ft Max'
    def derive(self, ils_glideslope=P('ILS Glideslope'),
               alt_aal = P('Altitude AAL'),
               estabs=S('ILS Glideslope Established')):
        for estab in estabs:
            for band in slices_above(alt_aal.array[estab.slice],1000)[1]:
                kpv_slice=[slice(estab.slice.start+band.start, estab.slice.start+band.stop)]
                self.create_kpvs_within_slices(ils_glideslope.array,kpv_slice,max_abs_value)  
            

class ILSGlideslopeDeviationBelow1000FtMax(KeyPointValueNode):
    name = 'ILS Glideslope Deviation Below 1000 Ft Max'
    def derive(self, ils_glideslope=P('ILS Glideslope'),
               alt_aal = P('Altitude AAL'),
               estabs=S('ILS Glideslope Established')):
        for estab in estabs:
            for band in slices_below(alt_aal.array[estab.slice],1000)[1]:
                kpv_slice=[slice(estab.slice.start+band.start, estab.slice.start+band.stop)]
                self.create_kpvs_within_slices(ils_glideslope.array,kpv_slice,max_abs_value)  

    
class ILSGlideslopeDeviation1000To150FtMax(KeyPointValueNode):
    name = 'ILS Glideslope Deviation 1000 To 150 Ft Max'
    def derive(self, ils_glideslope=P('ILS Glideslope'),
               alt_aal = P('Altitude AAL'),
               estabs=S('ILS Glideslope Established')):
        for estab in estabs:
            for band in slices_from_to(alt_aal.array[estab.slice],1000, 150)[1]:
                kpv_slice=[slice(estab.slice.start+band.start, estab.slice.start+band.stop)]
                self.create_kpvs_within_slices(ils_glideslope.array,kpv_slice,max_abs_value)  


class HeightAtGoAroundMin(KeyPointValueNode):
    def derive(self, alt=P('Altitude AAL'), go_around=KTI('Go Around')): 
        self.create_kpvs_at_ktis(alt.array, go_around)


class SinkRateWarning(KeyPointValueNode):
    def derive(self, taws_sink_rate=P('TAWS Sink Rate')):
        return NotImplemented


class HeadingDeviationOnTakeoffAbove100Kts(KeyPointValueNode):
    """
    The heading deviation is measured as the peak-to-peak deviation between
    100kts and 5 deg nose pitch up, at which time the weight is clearly off
    the wheel (we avoid using weight on nosewheel as this is often not
    recorded).
    The value is annotated half way between the end conditions.
    """
    def derive(self, head=P('Heading Continuous'), airspeed=P('Airspeed'),
               pitch=P('Pitch'), toffs=S('Takeoff')):
        for toff in toffs:
            start = index_at_value(airspeed.array, 100.0, _slice=toff.slice)
            stop = index_at_value(pitch.array, 5.0, _slice=toff.slice)
            head_dev = np.ma.ptp(head.array[start:stop])
            self.create_kpv((start+stop)/2, head_dev)
            

class HeadingDeviationOnLandingAbove100Kts(KeyPointValueNode):
    """
    See heading deviation on takeoff comments. For landing the Altitude AAL
    is used to detect start of landing, again to avoid variation from the use
    of different aircraft recording configurations.
    """
    def derive(self, head=P('Heading Continuous'), airspeed=P('Airspeed'),
               alt=P('Altitude AAL'), lands=S('Landing')):
        for land in lands:
            start = index_at_value(alt.array, 1.0, _slice=land.slice)
            stop = index_at_value(airspeed.array, 100.0, _slice=land.slice)
            head_dev = np.ma.ptp(head.array[start:stop])
            self.create_kpv((start+stop)/2, head_dev)
            

class HeightMinsToTouchdown(KeyPointValueNode):
    #TODO: TESTS
    #Q: Review and improve this technique of building KPVs on KTIs.
    from analysis_engine.key_time_instances import MinsToTouchdown
    NAME_FORMAT = "Height " + MinsToTouchdown.NAME_FORMAT
    NAME_VALUES = MinsToTouchdown.NAME_VALUES
    
    def derive(self, alt=P('Altitude AAL'), t_tdwns=KTI('Mins To Touchdown')):
        for t_tdwn in t_tdwns:
            #WARNING: This assumes Mins time will be the first value and only two digit
            # TODO: Confirm *.array is correct (DJ)
            self.create_kpv(t_tdwn.index, alt.array[t_tdwn.index], time=int(t_tdwn.name[:2]))
            
# See HeightMinsToTouchdown
##class Height1MinToTouchdown(KeyPointValueNode):
    ##def derive(self, altitude_aal=P('Altitude AAL'),
               ##_1_min_to_touchdown=KTI('1 Min To Touchdown')):
        ##return NotImplemented

# See HeightMinsToTouchdown
##class Height2MinToTouchdown(KeyPointValueNode):
    ##def derive(self, altitude_aal=P('Altitude AAL'),
               ##_2_min_to_touchdown=KTI('2 Min To Touchdown')):
        ##return NotImplemented


class HeightLost1000To2000FtMax(KeyPointValueNode):
    def derive(self, alt_aal=P('Altitude AAL')):
        return NotImplemented


class HeightLost50To1000Max(KeyPointValueNode):
    def derive(self, alt_aal=P('Altitude AAL')):
        return NotImplemented


class EngN13000FtToTouchdownMax(KeyPointValueNode):
    name = 'Eng N1 3000 Ft To Touchdown Max'
    def derive(self, eng=P('Eng (*) N1 Max'), alt_aal=P('Altitude AAL')):
        self.create_kpvs_within_slices(eng.array,
                                       alt_aal.slices_from_to(3000, 0),
                                       max_value)


class EngN1TakeoffMax(KeyPointValueNode):
    name = 'Eng N1 Takeoff Max'
    def derive(self, eng_n1_max=P('Eng (*) N1 Max'), ratings=S('Takeoff 5 Min Rating')):
        self.create_kpvs_within_slices(eng_n1_max.array, ratings, max_value)


class FlapAsGearSelectedDown(KeyPointValueNode):
    #NB: flap put as primary parameter as if we use Flap the step will be aligned!
    def derive(self, flap=P('Flap'), gear_sel_down=KTI('Gear Extending')):
        self.create_kpvs_at_ktis(flap.array, gear_sel_down)


class HeightAtConfigChange1500FtToTouchdownMin(KeyPointValueNode):
    def derive(self, flap=P('Flap'), alt_aal=P('Altitude AAL')):
        return NotImplemented
        #diff = np.ma.ediff1d(flap.array, to_begin=[0])
        #for this_slice in alt_aal.slices_from_to(1500, 0):
            #changes = alt_aal[diff][this_slice]
            #min(changes)
            #index, value = max_value(eng.array, this_slice)
            #self.create_kpv(index, value)


class SuspectedLevelBust(KeyPointValueNode):
    def derive(self, level_bust=S('Level Bust')):
        return NotImplemented


class ILSLocalizerDeviation1000To150FtMax(KeyPointValueNode):
    def derive(self, ils_loc=P('ILS Localizer'),
               alt_aal = P('Altitude AAL')):
        # For commented version, see GlideslopeDeviation1500To1000FtMax
        for this_period in alt_aal.slices_from_to(1000, 150):
            begin = this_period.start
            end = this_period.stop
            index_this_period = np.ma.argmax(np.ma.abs(ils_loc.array[begin:end]))
            index = begin + index_this_period
            value = ils_loc.array[index]
            self.create_kpv(index, value)


class ILSLocalizerDeviation1500To1000FtMax(KeyPointValueNode):
    def derive(self, ils_loc=P('ILS Localizer'),
               alt_aal = P('Altitude AAL')):
        # For commented version, see GlideslopeDeviation1500To1000FtMax
        for this_period in alt_aal.slices_from_to(1500, 1000):
            begin = this_period.start
            end = this_period.stop
            if alt_aal.array[begin] > alt_aal.array[end-1]:
                index_this_period = np.ma.argmax(np.ma.abs(ils_loc.array[begin:end]))
                index = begin + index_this_period
                value = ils_loc.array[index]
                self.create_kpv(index, value)


class Flare20FtToTouchdown(KeyPointValueNode):
    def derive(self, alt_rad=P('Altitude Radio'), alt_aal=P('Altitude AAL')):
        return NotImplemented
        


class LowPowerLessThan500Ft10Sec(KeyPointValueNode):
    #TODO: TESTS
    #Q: N1 Minimum or N1 Average
    def derive(self, eng_n1_min=P('Eng (*) N1 Min'), alt=P('Altitude AAL')):
        eng_clipped = clip(eng_n1_min.array, 10, eng_n1_min.hz)
        for alt_slice in alt.slices_from_to(500, 0):
            # clip to 10 secs
            index, value = min_value(eng_clipped, alt_slice)
            self.create_kpv(index, value)


class LowPowerInFinalApproach10Sec(KeyPointValueNode):
    #Q: N1 Minimum or N1 Average (as above)
    def derive(self, eng_avg=P('Eng (*) N1 Avg'), fin_apps=S('Final Approach')):
        return NotImplemented
    
    
class GroundspeedRTOMax(KeyPointValueNode):
    name = 'Groundspeed RTO Max'
    def derive(self, groundspeed=P('Groundspeed'),
               rejected_takeoff=S('Rejected Takeoff')):
        return NotImplemented


class EngN2Cycles(KeyPointValueNode):
    def derive(self, eng_n2=P('Eng (*) N2 Avg')):
        return NotImplemented


class EngOilPressMax(KeyPointValueNode):
    #TODO: TESTS
    name = 'Eng Oil Press Max'
    def derive(self, eng_oil_press=P('Eng (*) Oil Press Max')):
        index, value = max_value(eng_oil_press)
        self.create_kpv(index, value)
        

class EngOilPressMin(KeyPointValueNode):
    #TODO: TESTS
    name = 'Eng Oil Press Min'
    def derive(self, eng_oil_press=P('Eng (*) Oil Press Min')):
        index, value = min_value(eng_oil_press)
        self.create_kpv(index, value)


class FuelQtyAtLiftoff(KeyPointValueNode):
    def derive(self, fuel_qty=P('Fuel Qty'), liftoffs=KTI('Liftoff')):
        self.create_kpvs_at_ktis(fuel_qty.array, liftoffs)


class FuelQtyAtTouchdown(KeyPointValueNode):
    def derive(self, fuel_qty=P('Fuel Qty'), touchdowns=KTI('Touchdown')):
        self.create_kpvs_at_ktis(fuel_qty.array, touchdowns)


class GrossWeightAtLiftoff(KeyPointValueNode):
    def derive(self, gross_weight=P('Gross Weight Smoothed'), 
               liftoffs=KTI('Liftoff')):
        self.create_kpvs_at_ktis(gross_weight.array, liftoffs)


class GrossWeightAtTouchdown(KeyPointValueNode):
    def derive(self, gross_weight=P('Gross Weight Smoothed'),
               touchdowns=KTI('Touchdown')):
        self.create_kpvs_at_ktis(gross_weight.array, touchdowns)


class PitchCycles(KeyPointValueNode):
    #TODO: Review which period to reset the pitch cycles by (15 minute period?)
    def derive(self, pitch=P('Pitch')):
        return NotImplemented


class Pitch5FtToTouchdownMax(KeyPointValueNode):
    def derive(self, pitch=P('Pitch'), alt_aal=P('Altitude AAL')):
        self.create_kpvs_within_slices(pitch.array,
                                       alt_aal.slices_from_to(5, 0), max_value)


class PitchAtLiftoff(KeyPointValueNode):
    def derive(self, pitch=P('Pitch'), liftoffs=KTI('Liftoff')):
        self.create_kpvs_at_ktis(pitch.array, liftoffs)


class PitchDuringFinalApproachMin(KeyPointValueNode):
    def derive(self, pitch=P('Pitch'), final_approaches=S('Final Approach')):
        self.create_kpvs_within_slices(pitch.array, final_approaches, min_value)


class PitchDuringTakeoffMax(KeyPointValueNode):
    def derive(self, pitch=P('Pitch'), takeoffs=S('Takeoff')):
        self.create_kpvs_within_slices(pitch.array, takeoffs, max_value)


class PitchAtTouchdown(KeyPointValueNode):
    def derive(self, pitch=P('Pitch'), touchdowns=KTI('Touchdown')):
        self.create_kpvs_at_ktis(pitch.array, touchdowns)


class Pitch1000To100FtMax(KeyPointValueNode):
    def derive(self, pitch=P('Pitch'), alt_aal=P('Altitude AAL')):
        self.create_kpvs_within_slices(pitch.array,
                                       alt_aal.slices_from_to(1000, 100),
                                       max_value)


class Pitch1000To100FtMin(KeyPointValueNode):
    def derive(self, pitch=P('Pitch'), alt_aal=P('Altitude AAL')):
        self.create_kpvs_within_slices(pitch.array,
                                       alt_aal.slices_from_to(1000, 100),
                                       min_value)


class Pitch1000To500FtMin(KeyPointValueNode):
    def derive(self, pitch=P('Pitch'), alt_aal=P('Altitude AAL')):
        self.create_kpvs_within_slices(pitch.array,
                                       alt_aal.slices_from_to(1000, 500),
                                       min_value)


class Pitch500To100FtMin(KeyPointValueNode):
    def derive(self, pitch=P('Pitch'), alt_aal=P('Altitude AAL')):
        self.create_kpvs_within_slices(pitch.array,
                                       alt_aal.slices_from_to(500, 100),
                                       min_value)


class Pitch500To50FtMin(KeyPointValueNode):
    def derive(self, pitch=P('Pitch'), alt_aal=P('Altitude AAL')):
        self.create_kpvs_within_slices(pitch.array,
                                       alt_aal.slices_from_to(500, 50),
                                       min_value)


class Pitch20FtToTouchdownMin(KeyPointValueNode):
    def derive(self, pitch=P('Pitch'), alt_aal=P('Altitude AAL')):
        self.create_kpvs_within_slices(pitch.array,
                                       alt_aal.slices_from_to(20, 0), min_value)


class Pitch35To400FtMax(KeyPointValueNode):
    def derive(self, pitch=P('Pitch'), alt_aal=P('Altitude AAL')):
        self.create_kpvs_within_slices(pitch.array,
                                       alt_aal.slices_from_to(35, 400),
                                       max_value)


class Pitch35To400FtMin(KeyPointValueNode):
    def derive(self, pitch=P('Pitch'), alt_aal=P('Altitude AAL')):
        self.create_kpvs_within_slices(pitch.array,
                                       alt_aal.slices_from_to(35, 400),
                                       min_value)


class PitchRate35To1500FtMax(KeyPointValueNode):
    def derive(self, pitch_rate=P('Pitch Rate'), alt_aal=P('Altitude AAL')):
        self.create_kpvs_within_slices(pitch_rate.array,
                                       alt_aal.slices_from_to(35, 1500),
                                       max_value)


class PitchRateDuringTakeoffMin(KeyPointValueNode):
    def derive(self, pitch_rate=P('Pitch Rate'), takeoffs=S('Takeoff')):
        self.create_kpvs_within_slices(pitch_rate.array, takeoffs, min_value)


class PitchRateDuringTakeoffMax(KeyPointValueNode):
    def derive(self, pitch_rate=P('Pitch Rate'), takeoffs=S('Takeoff')):
        self.create_kpvs_within_slices(pitch_rate.array, takeoffs, max_value)


class PitchRateFrom2DegreesOfPitchDuringTakeoffMin(KeyPointValueNode):
    #TODO: TESTS
    def derive(self, pitch_rate=P('Pitch Rate'), pitch=P('Pitch'), 
               takeoffs=S('Takeoff')):
        for takeoff in takeoffs:
            # Endpoint closing allows for the case where the aircraft is at
            # more than 2 deg of pitch at takeoff.
            reversed_slice = slice(takeoff.slice.stop, takeoff.slice.start, -1)
            pitch_2_deg = index_at_value(pitch.array, 2, reversed_slice,
                                         endpoint='closing') #- takeoff.slice.start
            index, value = min_value(pitch_rate.array,
                                     slice(pitch_2_deg, takeoff.slice.stop))
            self.create_kpv(index, value)


class PowerOnWithSpeedbrakesDeployedDuration(KeyPointValueNode):
    def derive(self, eng_n1_average=P('Eng (*) N1 Avg'),
               speedbrake=P('Speedbrake')):
        return NotImplemented


class PullUpWarning(KeyPointValueNode):
    def derive(self, taws_pull_up=P('TAWS Pull Up')):
        return NotImplemented


class RollCycles1000FtToTouchdownMax(KeyPointValueNode):
    def derive(self, roll=P('Roll'), alt_aal=P('Altitude AAL')):
        return NotImplemented


class RollAbove1000FtMax(KeyPointValueNode):
    def derive(self, roll=P('Roll'), alt_aal=P('Altitude AAL')):
        self.create_kpvs_within_slices(roll.array, alt_aal.slices_above(1000),
                                       max_abs_value)

class RollAbove1500FtMax(KeyPointValueNode):
    def derive(self, roll=P('Roll'), alt_aal=P('Altitude AAL')):
        self.create_kpvs_within_slices(roll.array, alt_aal.slices_above(1500),
                                       max_abs_value)


class RollBelow20FtMax(KeyPointValueNode):
    def derive(self, roll=P('Roll'), alt_rad=P('Altitude Radio')):
        self.create_kpvs_within_slices(roll.array, alt_rad.slices_between(1,20),
                                       max_abs_value)


class RollBetween100And500FtMax(KeyPointValueNode):
    def derive(self, roll=P('Roll'), alt_aal=P('Altitude AAL')):
        self.create_kpvs_within_slices(roll.array,
                                       alt_aal.slices_between(100, 500),
                                       max_abs_value)


class RollBetween500And1500FtMax(KeyPointValueNode):
    def derive(self, roll=P('Roll'), alt_aal=P('Altitude AAL')):
        self.create_kpvs_within_slices(roll.array,
                                       alt_aal.slices_between(500, 1500),
                                       max_abs_value)


class RudderReversalAbove50Ft(KeyPointValueNode):
    def derive(self, rudder_reversal=P('Rudder Reversal'),
               alt_aal=P('Altitude AAL')):
        # Q: Should this be Max or Min?
        return NotImplemented


class TaxiSpeedTurning(KeyPointValueNode):
    def derive(self, gspd=P('Groundspeed'), taxis=S('Taxi'), rot=P('Rate Of Turn')):
        for taxi in taxis:
            turns = np.ma.clump_masked(np.ma.masked_greater(rot.array[taxi.slice], RATE_OF_TURN_FOR_TAXI_TURNS))
            for turn in turns:
                index = np.ma.argmax(gspd.array[taxi.slice][turn])
                value = gspd.array[taxi.slice][turn][index]
                self.create_kpv(int(taxi.slice.start)+turn+index, value)


class TaxiSpeedStraight(KeyPointValueNode):
    def derive(self, gspd=P('Groundspeed'), taxis=S('Taxi'), rot=P('Rate Of Turn')):
        for taxi in taxis:
            straights = np.ma.clump_unmasked(np.ma.masked_greater(rot.array[taxi.slice], RATE_OF_TURN_FOR_TAXI_TURNS))
            for straight in straights:
                index = np.ma.argmax(gspd.array[taxi.slice][straight])
                value = gspd.array[taxi.slice][straight][index]
                self.create_kpv(int(taxi.slice.start)+straight+index, value)


class TaxiSpeed(KeyPointValueNode):
    def derive(self, gspd=P('Groundspeed'), taxis=S('Taxi')):
        for taxi in taxis:
            index = np.ma.argmax(gspd.array[taxi.slice])
            value = gspd.array[taxi.slice][index]
            self.create_kpv(int(taxi.slice.start)+index, value)


class AirspeedMinusVref500FtToTouchdownMax(KeyPointValueNode):
    def derive(self, airspeed_minus_vref=P('Airspeed Minus Vref'),
               alt_aal=P('Altitude AAL')):
        self.create_kpvs_within_slices(airspeed_minus_vref.array,
                                       alt_aal.slices_from_to(500, 0),
                                       max_value)


class AirspeedWithFlapMax(KeyPointValueNode):
    NAME_FORMAT = "Airspeed With Flap %(flap)d Max"
    NAME_VALUES = {'flap': range(0,46,1)}
    #Q: Is it required to have a KPV of "Flap 0 Max"
    def derive(self, airspeed=P('Airspeed'), flap=P('Flap')):
        for flap_setting in np.ma.unique(flap.array):
            if np.ma.is_masked(flap_setting):
                # ignore masked values
                continue
            spd_with_flap = np.ma.copy(airspeed.array)
            # apply flap mask
            spd_with_flap.mask = np.ma.mask_or(airspeed.array.mask, flap.array.mask)
            spd_with_flap[flap.array != flap_setting] = np.ma.masked
            #TODO: Check logical OR is sensible for all values (probably ok as airspeed will always be higher than max flap setting!)
            index, value = max_value(spd_with_flap)
            self.create_kpv(index, value, flap=flap_setting)


class AirspeedMinusVrefAtTouchdown(KeyPointValueNode):
    def derive(self, airspeed=P('Airspeed Minus Vref'), tdwns=KTI('Touchdown')):
        self.create_kpvs_at_ktis(airspeed.array, tdwns)


class AirspeedBetween90SecToTouchdownAndTouchdownMax(KeyPointValueNode):
    def derive(self, sec_to_touchdown=KTI('Secs To Touchdown'), airspeed=P('Airspeed')):
        for _90_sec in sec_to_touchdown.get(name='90 Secs To Touchdown'):
            # we're 90 seconds from touchdown
            tdwn = _90_sec.index + 90 * self.frequency
            index, value = max_value(airspeed.array, slice(_90_sec.index, tdwn))
            self.create_kpv(index, value)


class AirspeedMinusV2AtLiftoff(KeyPointValueNode):
    name = 'Airspeed Minus V2 At Liftoff'
    def derive(self, airspeed=P('Airspeed Minus V2'), liftoffs=KTI('Liftoff')):
        self.create_kpvs_at_ktis(airspeed.array, liftoffs)
        
        
class AirspeedMinusV235To400FtMin(KeyPointValueNode):
    name = 'Airspeed Minus V2 35 To 400 Ft Min'
    def derive(self, airspeed=P('Airspeed Minus V2'), alt_aal=P('Altitude AAL')):
        self.create_kpvs_within_slices(airspeed.array,
                                       alt_aal.slices_from_to(35, 400),
                                       min_value)


class AirspeedMinusV2400To1500FtMin(KeyPointValueNode):
    name = 'Airspeed Minus V2 400 To 1500 Ft Min'
    def derive(self, airspeed=P('Airspeed Minus V2'), alt_aal=P('Altitude AAL')):
        self.create_kpvs_within_slices(airspeed.array,
                                       alt_aal.slices_from_to(400, 1500),
                                       min_value)


class AirspeedMinusVrefBetween2MinutesToTouchdownAndTouchdownMin(KeyPointValueNode):
    #TODO: TESTS
    def derive(self, mins_to_touchdown=KTI('Mins To Touchdown'), 
               airspeed=P('Airspeed Minus Vref')):
        for _2_min in mins_to_touchdown.get(name='2 Mins To Touchdown'):
            # add 2 mins to find touchdown
            tdwn = _2_min.index + 2 * 60 * self.frequency
            index, value = min_value(airspeed.array, slice(_2_min.index, tdwn))
            self.create_kpv(index, value)

            
class SpeedbrakesDeployed1000To25Ft(KeyPointValueNode):
    def derive(self, speedbrake=P('Speedbrake'), alt_aal=P('Altitude AAL')):
        return NotImplemented


class FlapWithSpeedbrakesDeployedMax(KeyPointValueNode):
    #TODO: TESTS
    def derive(self, flap=P('Flap'), speedbrake=P('Speedbrake')):
        # mask all values where speedbrake isn't deployed
        # assumes that Speedbrake == 0 is not_deployed
        flap.array[speedbrake.array == 0] = np.ma.masked
        index, value = max_value(flap.array)
        self.create_kpv(index, value)


class StickShakerActivated(KeyPointValueNode):
    def derive(self, stick_shaker=P('Stick Shaker')):
        return NotImplemented


class TAWSTerrainWarning(KeyPointValueNode):
    name = 'TAWS Terrain Warning'
    def derive(self, taws_terrain=P('TAWS Terrain')):
        return NotImplemented


class TAWSTerrainPullUpWarning(KeyPointValueNode):
    name = 'TAWS Terrain Pull Up Warning'
    def derive(self, taws_terrain_pull_up=P('TAWS Terrain Pull Up')):
        return NotImplemented


class ThrottleCycles1000FtToTouchdownMax(KeyPointValueNode):
    def derive(self, thrust_lever=P('Throttle Lever')):
        return NotImplemented


class TooLowFlapWarning(KeyPointValueNode):
    def derive(self, taws_too_low_flap=P('TAWS Too Low Flap'), flap=P('Flap')):
        return NotImplemented


class TooLowGearWarning(KeyPointValueNode):
    def derive(self, taws_too_low_gear=P('TAWS Too Low Gear')):
        return NotImplemented


class TAWSTooLowTerrainWarning(KeyPointValueNode):
    name = 'TAWS Too Low Terrain Warning'
    def derive(self, taws_too_low_terrain=P('TAWS Too Low Terrain')):
        slices = slices_above(taws_too_low_terrain.array, 1)[1]
        for too_low_terrain_slice in slices:
            index = too_low_terrain_slice.start
            value = taws_too_low_terrain.array[too_low_terrain_slice.start]
            self.create_kpv(index, value)


class EngVibN1Above_XXXXX_Duration(KeyPointValueNode):
    def derive(self, eng_n_vib_n1=P('Eng (*) Vib N1 Max'),
               airborne=S('Airborne')):
        return NotImplemented


class EngVibN2Above_XXXXX_Duration(KeyPointValueNode):
    def derive(self, eng_vib_n2=P('Eng (*) Vib N2 Max'),
               airborne=S('Airborne')):
        return NotImplemented


class WindshearWarningBelow1500Ft(KeyPointValueNode):
    def derive(self, taws_windshear=P('TAWS Windshear'),
               alt_aal=P('Altitude AAL')):
        return NotImplemented


class AirspeedBelowAltitudeMax(KeyPointValueNode):
    NAME_FORMAT = 'Airspeed Below %(altitude)d Ft Max'
    NAME_VALUES = {'altitude': [500, 3000, 7000]}
    
    def derive(self, airspeed=P('Airspeed'), alt_aal=P('Altitude AAL')):
        for alt in self.NAME_VALUES['altitude']:
            self.create_kpvs_within_slices(airspeed.array,
                                           alt_aal.slices_below(alt),
                                           max_value,
                                           altitude=alt)

class AirspeedBelowFL100Max(KeyPointValueNode):
    '''
    TODO: Test.
    '''
    name = 'Airspeed Below FL100 Max'
    def derive(self, alt_std=P('Altitude STD'), airspeed=P('Airspeed'),
               in_airs=S('Airborne')):
        # Other airspeed tests relate to heights above the runway, whereas
        # this is flight level dependent. Altitude_AAL is invalid at low
        # speeds, whereas alt_std is always valid, hence why the conditional
        # airborne element is required.
        for in_air in in_airs:
            self.create_kpvs_within_slices(airspeed.array,
                                           alt_std.slices_below(10000),
                                           max_value)

class AltitudeRadioDividedByDistanceToLanding3000To50FtMin(KeyPointValueNode):
    def derive(self, alt_aal=P('Altitude AAL'), alt_radio=P('Altitude Radio'), 
               dtl=P('Distance To Landing')):
        for desc_slice in alt_aal.slices_from_to(3000, 50):
            angle_array = alt_radio.array[desc_slice]/dtl.array[desc_slice]
            
    
class TailClearanceOnApproach(KeyPointValueNode):
    def derive(self, alt_aal=P('Altitude AAL'), alt_tail=P('Altitude Tail'), 
                 dtl=P('Distance To Landing')):
        '''
        This finds abnormally low tail clearance during the approach down to
        100ft. It searches for the minimum angular separation between the
        flightpath and the terrain, so a 500ft clearance at 2500ft AAL is
        considered more significant than 500ft at 1500ft AAL. The value
        stored is the tail clearance. A matching KTI will allow these to be
        located on the approach chart.
        '''
        for desc_slice in alt_aal.slices_from_to(3000, 100):
            angle_array = alt_tail.array[desc_slice]/dtl.array[desc_slice]
            index, value = min_value(angle_array)
            if index:
                sample = index + desc_slice.start
                self.create_kpv(sample, alt_tail.array[sample])
    
    
class ZeroFuelWeight(KeyPointValueNode):
    """
    The aircraft zero fuel weight is estimated from the recorded gross weight
    and fuel data. A best fit line should be close to a straight line, with a
    slope of 1.0 and the intercept at fuel=0 is the Zero Fuel Weight. The
    test for slope is important as this traps errors where one of the weights
    is converted in the wrong units!
    
    See also the GrossWeightSmoothed calculation which uses fuel flow data to
    obtain a higher sample rate solution to the aircraft weight calculation.
    """
    def derive(self, fuel=P('Fuel Qty'), gw=P('Gross Weight')):
        corr, slope, zfw = coreg(gw.array, indep_var=fuel.array)
        if corr>0.9 and 0.8 < slope < 1.2:
            self.create_kpv(0, zfw)
        else:
            raise ValueError, 'Unable to compute Zero Fuel Weight from data supplied'
            
        
        
