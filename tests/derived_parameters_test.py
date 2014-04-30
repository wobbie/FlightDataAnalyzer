import numpy as np

import datetime
import os
import shutil
import sys
import tempfile
import unittest

from itertools import izip
from mock import Mock, call, patch

from hdfaccess.file import hdf_file
from flightdatautilities import aircrafttables as at, units as ut
from flightdatautilities.aircrafttables.interfaces import VelocitySpeed
from flightdatautilities import masked_array_testutils as ma_test
from flightdatautilities.filesystem_tools import copy_file

from analysis_engine.flight_phase import Fast, Mobile, RejectedTakeoff
from analysis_engine.library import (align,
                                     max_value,
                                     np_ma_masked_zeros,
                                     np_ma_masked_zeros_like,
                                     np_ma_ones_like,
                                     unique_values)

from analysis_engine.node import (Attribute,
                                  A,
                                  App,
                                  ApproachItem,
                                  KeyPointValue,
                                  KPV,
                                  KeyTimeInstance,
                                  KTI,
                                  load,
                                  M,
                                  Parameter,
                                  P,
                                  Section,
                                  S)
from analysis_engine.process_flight import process_flight
from analysis_engine.settings import GRAVITY_IMPERIAL, METRES_TO_FEET

from flight_phase_test import buildsection, buildsections

from analysis_engine.derived_parameters import (
    #ATEngaged,
    AccelerationLateralSmoothed,
    AccelerationVertical,
    AccelerationForwards,
    AccelerationSideways,
    AccelerationAlongTrack,
    AccelerationAcrossTrack,
    Aileron,
    AimingPointRange,
    AirspeedForFlightPhases,
    AirspeedTrue,
    AltitudeAAL,
    AltitudeAALForFlightPhases,
    #AltitudeForFlightPhases,
    AltitudeQNH,
    AltitudeRadio,
    AltitudeSTDSmoothed,
    #AltitudeRadioForFlightPhases,
    #AltitudeSTD,
    AltitudeTail,
    AOA,
    ApproachRange,
    BrakePressure,
    CabinAltitude,
    ClimbForFlightPhases,
    ControlColumn,
    ControlColumnForce,
    ControlWheel,
    ControlWheelForce,
    CoordinatesSmoothed,
    DescendForFlightPhases,
    DistanceTravelled,
    DistanceToLanding,
    Drift,
    Elevator,
    ElevatorLeft,
    ElevatorRight,
    Eng_EPRAvg,
    Eng_EPRMax,
    Eng_EPRMin,
    Eng_EPRMinFor5Sec,
    Eng_FuelFlow,
    Eng_FuelFlowMax,
    Eng_FuelFlowMin,
    Eng_N1Avg,
    Eng_N1Max,
    Eng_N1Min,
    Eng_N1MinFor5Sec,
    Eng_N2Avg,
    Eng_N2Max,
    Eng_N2Min,
    Eng_N3Avg,
    Eng_N3Max,
    Eng_N3Min,
    Eng_NpAvg,
    Eng_NpMax,
    Eng_NpMin,
    Eng_TorquePercentAvg,
    Eng_TorquePercentMax,
    Eng_TorquePercentMin,
    Eng_VibBroadbandMax,
    Eng_VibN1Max,
    Eng_VibN2Max,
    Eng_VibN3Max,
    Eng_VibAMax,
    Eng_VibBMax,
    Eng_VibCMax,
    Eng_1_FuelBurn,
    Eng_2_FuelBurn,
    Eng_3_FuelBurn,
    Eng_4_FuelBurn,
    EngTPRLimitDifference,
    FlapAngle,
    FuelQty,
    FuelQtyL,
    FuelQtyR,
    GrossWeight,
    GrossWeightSmoothed,
    Groundspeed,
    #GroundspeedAlongTrack,
    Heading,
    HeadingContinuous,
    HeadingIncreasing,
    HeadingTrue,
    Headwind,
    ILSFrequency,
    #ILSGlideslope,
    #ILSLocalizer,
    #ILSLocalizerRange,
    LatitudePrepared,
    LatitudeSmoothed,
    LongitudePrepared,
    LongitudeSmoothed,
    Mach,
    MagneticVariation,
    MagneticVariationFromRunway,
    Pitch,
    RollRate,
    RudderPedal,
    SidestickAngleCapt,
    SidestickAngleFO,
    SlatAngle,
    Speedbrake,
    Stabilizer,
    VerticalSpeed,
    VerticalSpeedForFlightPhases,
    RateOfTurn,
    Roll,
    Rudder,
    ThrottleLevers,
    ThrustAsymmetry,
    TrackDeviationFromRunway,
    Track,
    TrackContinuous,
    TrackTrue,
    TrackTrueContinuous,
    TurbulenceRMSG,
    VerticalSpeedInertial,
    WheelSpeed,
    WheelSpeedLeft,
    WheelSpeedRight,
    WindAcrossLandingRunway,
    WindDirection,
    WindDirectionTrue,
    # Velocity Speeds:
    AirspeedMinusFlapManoeuvreSpeed,
    AirspeedMinusFlapManoeuvreSpeedFor3Sec,
    AirspeedMinusMinimumAirspeed,
    AirspeedMinusMinimumAirspeedFor3Sec,
    AirspeedMinusV2,
    AirspeedMinusV2For3Sec,
    AirspeedMinusVapp,
    AirspeedMinusVappFor3Sec,
    AirspeedMinusVref,
    AirspeedMinusVrefFor3Sec,
    AirspeedRelative,
    AirspeedRelativeFor3Sec,
    FlapManoeuvreSpeed,
    MMOLookup,
    MinimumAirspeed,
    V2,
    V2Lookup,
    Vapp,
    VappLookup,
    Vref,
    VrefLookup,
    VMOLookup,
    ZeroFuelWeight,
)


##############################################################################
# Test Configuration


def setUpModule():
    at.configure(package='flightdatautilities.aircrafttables')


##############################################################################


debug = sys.gettrace() is not None

test_data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              'test_data')

def assert_array_within_tolerance(actual, desired, tolerance=1, similarity=100):
    '''
    Check that the actual array within tolerance of the desired array is
    at least similarity percent.
    
    :param tolerance: relative difference between the two array values
    :param similarity: percentage that must pass the tolerance test
    '''
    within_tolerance = abs(actual -  desired) <= tolerance
    percent_similar = np.ma.sum(within_tolerance) / float(len(within_tolerance)) * 100
    if percent_similar <= similarity:
        raise AssertionError(
            'actual array tolerance only is %.2f%% similar to desired array.'
            'tolerance %.2f minimum similarity required %.2f%%' % (
                percent_similar, tolerance, similarity))


class TemporaryFileTest(object):
    '''
    Test using a temporary copy of a predefined file.
    '''
    def setUp(self):
        if getattr(self, 'source_file_path', None):
            self.make_test_copy()

    def tearDown(self):
        if self.test_file_path:
            os.unlink(self.test_file_path)
            self.test_file_path = None

    def make_test_copy(self):
        '''
        Copy the test file to temporary location, used by setUp().
        '''
        # Create the temporary file in the most secure way
        f = tempfile.NamedTemporaryFile(delete=False)
        self.test_file_path = f.name
        f.close()
        shutil.copy2(self.source_file_path, self.test_file_path)


class NodeTest(object):

    def generate_attributes(self, manufacturer):
        if manufacturer == 'boeing':
            _am = A('Model', 'B737-333')
            _as = A('Series', 'B737-300')
            _af = A('Family', 'B737 Classic')
            _et = A('Engine Type', 'CFM56-3B1')
            _es = A('Engine Series', 'CFM56-3')
            return (_am, _as, _af, _et, _es)
        if manufacturer == 'airbus':
            _am = A('Model', 'A330-333')
            _as = A('Series', 'A330-300')
            _af = A('Family', 'A330')
            _et = A('Engine Type', 'Trent 772B-60')
            _es = A('Engine Series', 'Trent 772B')
            return (_am, _as, _af, _et, _es)
        if manufacturer == 'beechcraft':
            _am = A('Model', '1900D')
            _as = A('Series', '1900D')
            _af = A('Family', '1900')
            _et = A('Engine Type', 'PT6A-67D')
            _es = A('Engine Series', 'PT6A')
            return (_am, _as, _af, _et, _es)
        raise ValueError('Unexpected lookup for attributes.')

    def test_can_operate(self):
        if getattr(self, 'check_operational_combination_length_only', False):
            self.assertEqual(
                len(self.node_class.get_operational_combinations()),
                self.operational_combination_length,
            )
        else:
            combinations = map(set, self.node_class.get_operational_combinations())
            for combination in map(set, self.operational_combinations):
                self.assertIn(combination, combinations)

    def get_params_from_hdf(self, hdf_path, param_names, _slice=None,
                            phase_name='Phase'):
        import shutil
        import tempfile

        params = []
        phase = None

        with tempfile.NamedTemporaryFile() as temp_file:
            shutil.copy(hdf_path, temp_file.name)

            with hdf_file(hdf_path) as hdf:
                for param_name in param_names:
                    params.append(hdf.get(param_name))

        if _slice:
            phase = S(name=phase_name, frequency=1)
            phase.create_section(_slice)
            phase = phase.get_aligned(params[0])

        return params, phase


##### FIXME: Re-enable when 'AT Engaged' has been implemented.
####class TestATEngaged(unittest.TestCase, NodeTest):
####
####    def setUp(self):
####        self.node_class = ATEngaged
####        self.operational_combinations = [
####            ('AT (1) Engaged',),
####            ('AT (2) Engaged',),
####            ('AT (3) Engaged',),
####            ('AT (1) Engaged', 'AT (2) Engaged'),
####            ('AT (1) Engaged', 'AT (3) Engaged'),
####            ('AT (2) Engaged', 'AT (3) Engaged'),
####            ('AT (1) Engaged', 'AT (2) Engaged', 'AT (3) Engaged'),
####        ]
####
####    @unittest.skip('Test Not Implemented')
####    def test_derive(self):
####        self.assertTrue(False, msg='Test not implemented.')


##############################################################################


class TestAccelerationLateralSmoothed(unittest.TestCase):
    def test_can_operate(self):
        opts = AccelerationLateralSmoothed.get_operational_combinations()
        self.assertEqual(opts, [('Acceleration Lateral Offset Removed',)])
        
    def test_smoothing(self):
        acc = AccelerationLateralSmoothed()
        acc.derive(P(array=np.ma.array([0]*20 + [0, 0, 0, 0, 0, 0, 5, 10, 10, 5, 
                                        25, 10, 5, 5, 0]),
                     frequency=4))
        self.assertEqual(acc.window, 9)  # 2secs * 4hz +1
        self.assertEqual(np.ma.min(acc.array), 0)
        self.assertAlmostEqual(np.ma.max(acc.array), 8.333, 2)
                         
    


class TestAccelerationVertical(unittest.TestCase):
    def test_can_operate(self):
        expected = [('Acceleration Normal Offset Removed',
                     'Acceleration Lateral Offset Removed',
                     'Acceleration Longitudinal', 'Pitch', 'Roll')]
        opts = AccelerationVertical.get_operational_combinations()
        self.assertEqual(opts, expected)
        
    def test_acceleration_vertical_level_on_gound(self):
        # Invoke the class object
        acc_vert = AccelerationVertical(frequency=8)
                        
        acc_vert.get_derived([
            Parameter('Acceleration Normal Offset Removed', np.ma.ones(8), 8),
            Parameter('Acceleration Lateral Offset Removed', np.ma.zeros(4), 4),
            Parameter('Acceleration Longitudinal', np.ma.zeros(4), 4),
            Parameter('Pitch', np.ma.zeros(2), 2),
            Parameter('Roll', np.ma.zeros(2), 2),
        ])
        
        #                                     x   interp  x  pitch/roll masked
        expected = np.ma.array([1] * 8, mask=[0, 0, 0, 0, 0,   1, 1, 1])
        ma_test.assert_masked_array_approx_equal(acc_vert.array, expected)
        
    def test_acceleration_vertical_pitch_up(self):
        acc_vert = AccelerationVertical(frequency=8)

        acc_vert.get_derived([
            P('Acceleration Normal Offset Removed',np.ma.ones(8) * 0.8660254,8),
            P('Acceleration Lateral Offset Removed',np.ma.zeros(4), 4),
            P('Acceleration Longitudinal',np.ma.ones(4) * 0.5,4),
            P('Pitch',np.ma.ones(2) * 30.0,2),
            P('Roll',np.ma.zeros(2), 2)
        ])

        #                                     x   interp  x  pitch/roll masked
        expected = np.ma.array([1] * 8, mask=[0, 0, 0, 0, 0,   1, 1, 1])
        ma_test.assert_masked_array_approx_equal(acc_vert.array, expected)

    def test_acceleration_vertical_pitch_up_roll_right(self):
        acc_vert = AccelerationVertical(frequency=8)

        acc_vert.get_derived([
            P('Acceleration Normal Offset Removed', np.ma.ones(8) * 0.8, 8),
            P('Acceleration Lateral Offset Removed', np.ma.ones(4) * (-0.2), 4),
            P('Acceleration Longitudinal', np.ma.ones(4) * 0.3, 4),
            P('Pitch',np.ma.ones(2) * 30.0, 2),
            P('Roll',np.ma.ones(2) * 20, 2)])
        
        expected = np.ma.array([0.86027777] * 8, mask=[0, 0, 0, 0, 0,   1, 1, 1])
        ma_test.assert_masked_array_approx_equal(acc_vert.array, expected)

    def test_acceleration_vertical_roll_right(self):
        acc_vert = AccelerationVertical(frequency=8)

        acc_vert.get_derived([
            P('Acceleration Normal Offset Removed', np.ma.ones(8) * 0.7071068, 8),
            P('Acceleration Lateral Offset Removed', np.ma.ones(4) * -0.7071068, 4),
            P('Acceleration Longitudinal', np.ma.zeros(4), 4),
            P('Pitch', np.ma.zeros(2), 2),
            P('Roll', np.ma.ones(2) * 45, 2),
        ])
        #                                     x   interp  x  pitch/roll masked
        expected = np.ma.array([1] * 8, mask=[0, 0, 0, 0, 0,   1, 1, 1])
        ma_test.assert_masked_array_approx_equal(acc_vert.array, expected)


class TestAccelerationForwards(unittest.TestCase):
    def test_can_operate(self):
        expected = [('Acceleration Normal Offset Removed',
                    'Acceleration Longitudinal', 'Pitch')]
        opts = AccelerationForwards.get_operational_combinations()
        self.assertEqual(opts, expected)
        
    def test_acceleration_forward_level_on_gound(self):
        # Invoke the class object
        acc_fwd = AccelerationForwards(frequency=4)
                        
        acc_fwd.get_derived([
            Parameter('Acceleration Normal Offset Removed', np.ma.ones(8), 8),
            Parameter('Acceleration Longitudinal', np.ma.ones(4) * 0.1,4),
            Parameter('Pitch', np.ma.zeros(2), 2)
        ])
        expected = np.ma.array([0.1] * 8, mask=[0, 0, 0, 0, 0,   1, 1, 1])
        ma_test.assert_masked_array_approx_equal(acc_fwd.array, expected)
        
    def test_acceleration_forward_pitch_up(self):
        acc_fwd = AccelerationForwards(frequency=4)

        acc_fwd.get_derived([
            P('Acceleration Normal Offset Removed', np.ma.ones(8) * 0.8660254, 8),
            P('Acceleration Longitudinal', np.ma.ones(4) * 0.5, 4),
            P('Pitch', np.ma.ones(2) * 30.0, 2)
        ])

        expected = np.ma.array([0] * 8, mask=[0, 0, 0, 0, 0,   1, 1, 1])
        ma_test.assert_masked_array_approx_equal(acc_fwd.array, expected)


class TestAccelerationSideways(unittest.TestCase):
    def test_can_operate(self):
        expected = [('Acceleration Normal Offset Removed',
                    'Acceleration Lateral Offset Removed', 
                    'Acceleration Longitudinal', 'Pitch', 'Roll')]
        opts = AccelerationSideways.get_operational_combinations()
        self.assertEqual(opts, expected)
        
    def test_acceleration_sideways_level_on_gound(self):
        # Invoke the class object
        acc_lat = AccelerationSideways(frequency=8)
                        
        acc_lat.get_derived([
            Parameter('Acceleration Normal Offset Removed', np.ma.ones(8),8),
            Parameter('Acceleration Lateral Offset Removed', np.ma.ones(4)*0.05,4),
            Parameter('Acceleration Longitudinal', np.ma.zeros(4),4),
            Parameter('Pitch', np.ma.zeros(2),2),
            Parameter('Roll', np.ma.zeros(2),2)
        ])
        #                                     x   interp  x  pitch/roll masked
        expected = np.ma.array([0.05] * 8, mask=[0, 0, 0, 0, 0,   1, 1, 1])
        ma_test.assert_masked_array_approx_equal(acc_lat.array, expected)
        
    def test_acceleration_sideways_pitch_up(self):
        acc_lat = AccelerationSideways(frequency=8)

        acc_lat.get_derived([
            P('Acceleration Normal Offset Removed',np.ma.ones(8)*0.8660254,8),
            P('Acceleration Lateral Offset Removed',np.ma.zeros(4),4),
            P('Acceleration Longitudinal',np.ma.ones(4)*0.5,4),
            P('Pitch',np.ma.ones(2)*30.0,2),
            P('Roll',np.ma.zeros(2),2)
        ])
        #                                     x   interp  x  pitch/roll masked
        expected = np.ma.array([0] * 8, mask=[0, 0, 0, 0, 0,   1, 1, 1])
        ma_test.assert_masked_array_approx_equal(acc_lat.array, expected)

    def test_acceleration_sideways_roll_right(self):
        acc_lat = AccelerationSideways(frequency=8)

        acc_lat.get_derived([
            P('Acceleration Normal Offset Removed',np.ma.ones(8)*0.7071068,8),
            P('Acceleration Lateral Offset Removed',np.ma.ones(4)*(-0.7071068),4),
            P('Acceleration Longitudinal',np.ma.zeros(4),4),
            P('Pitch',np.ma.zeros(2),2),
            P('Roll',np.ma.ones(2)*45,2)
        ])
        #                                     x   interp  x  pitch/roll masked
        expected = np.ma.array([0] * 8, mask=[0, 0, 0, 0, 0,   1, 1, 1])
        ma_test.assert_masked_array_approx_equal(acc_lat.array, expected)

        
class TestAccelerationAcrossTrack(unittest.TestCase):
    def test_can_operate(self):
        expected = [('Acceleration Forwards',
                    'Acceleration Sideways', 'Drift')]
        opts = AccelerationAcrossTrack.get_operational_combinations()
        self.assertEqual(opts, expected)
        
    def test_acceleration_across_side_only(self):
        acc_across = AccelerationAcrossTrack()
        acc_across.get_derived([
            Parameter('Acceleration Forwards', np.ma.ones(8), 8),
            Parameter('Acceleration Sideways', np.ma.ones(4)*0.1, 4),
            Parameter('Drift', np.ma.zeros(2), 2)])
        expected = np.ma.array([0.1] * 8, mask=[0, 0, 0, 0, 0,   1, 1, 1])
        ma_test.assert_masked_array_approx_equal(acc_across.array, expected)
        
    def test_acceleration_across_resolved(self):
        acc_across = AccelerationAcrossTrack()
        acc_across.get_derived([
            P('Acceleration Forwards',np.ma.ones(8)*0.8660254,8),
            P('Acceleration Sideways',np.ma.ones(4)*0.5,4),
            P('Drift',np.ma.ones(2)*30.0,2)])

        expected = np.ma.array([0] * 8, mask=[0, 0, 0, 0, 0,   1, 1, 1])
        ma_test.assert_masked_array_approx_equal(acc_across.array, expected)

class TestAccelerationAlongTrack(unittest.TestCase):
    def test_can_operate(self):
        expected = [('Acceleration Forwards',
                    'Acceleration Sideways', 'Drift')]
        opts = AccelerationAlongTrack.get_operational_combinations()
        self.assertEqual(opts, expected)
        
    def test_acceleration_along_forward_only(self):
        acc_along = AccelerationAlongTrack()
        acc_along.get_derived([
            Parameter('Acceleration Forwards', np.ma.ones(8)*0.2,8),
            Parameter('Acceleration Sideways', np.ma.ones(4)*0.1,4),
            Parameter('Drift', np.ma.zeros(2),2)])
        
        expected = np.ma.array([0.2] * 8, mask=[0, 0, 0, 0, 0,   1, 1, 1])
        ma_test.assert_masked_array_approx_equal(acc_along.array, expected)
        
    def test_acceleration_along_resolved(self):
        acc_along = AccelerationAlongTrack()
        acc_along.get_derived([
            P('Acceleration Forwards',np.ma.ones(8)*0.1,8),
            P('Acceleration Sideways',np.ma.ones(4)*0.2,4),
            P('Drift',np.ma.ones(2)*10.0,2)])
        expected = np.ma.array([0.13321041] * 8, mask=[0, 0, 0, 0, 0,   1, 1, 1])
        ma_test.assert_masked_array_approx_equal(acc_along.array, expected)


class TestAirspeedForFlightPhases(unittest.TestCase):
    def test_can_operate(self):
        expected = [('Airspeed',)]
        opts = AirspeedForFlightPhases.get_operational_combinations()
        self.assertEqual(opts, expected)


class TestAirspeedTrue(unittest.TestCase):
    def test_can_operate(self):
        self.assertIn(('Airspeed', 'Altitude STD'), AirspeedTrue.get_operational_combinations())
        self.assertIn(('Airspeed', 'Altitude STD', 'SAT', 
                       'Takeoff', 'Landing', 'Rejected Takeoff', 
                       'Groundspeed', 'Acceleration Forwards'), 
                      AirspeedTrue.get_operational_combinations())
        
    def test_tas_basic(self):
        cas = P('Airspeed', np.ma.array([100, 200, 300]))
        alt = P('Altitude STD', np.ma.array([0, 20000, 40000]))
        sat = P('SAT', np.ma.array([20, -10, -55]))
        tas = AirspeedTrue()
        tas.derive(cas, alt, sat)
        result = [100.864, 278.375, 555.595]
        self.assertLess(abs(tas.array.data[0] - result[0]), 0.01)
        self.assertLess(abs(tas.array.data[1] - result[1]), 0.01)
        self.assertLess(abs(tas.array.data[2] - result[2]), 0.01)
        
    def test_tas_masks(self):
        cas = P('Airspeed', np.ma.array([100, 200, 300]))
        alt = P('Altitude STD', np.ma.array([0, 20000, 40000]))
        tat = P('TAT', np.ma.array([20, -10, -40]))
        tas = AirspeedTrue()
        cas.array[0] = np.ma.masked
        alt.array[1] = np.ma.masked
        tat.array[2] = np.ma.masked
        tas.derive(cas, alt, tat)
        np.testing.assert_array_equal(tas.array.mask, [True] * 3)
        
    def test_tas_no_tat(self):
        cas = P('Airspeed', np.ma.array([100, 200, 300]))
        alt = P('Altitude STD', np.ma.array([0, 10000, 20000]))
        tas = AirspeedTrue()
        tas.derive(cas, alt, None)
        result = [100.000, 231.575, 400.097]
        self.assertLess(abs(tas.array.data[0] - result[0]), 0.01)
        self.assertLess(abs(tas.array.data[1] - result[1]), 0.01)
        self.assertLess(abs(tas.array.data[2] - result[2]), 0.01)
        
    def test_tas_with_gs_extensions(self):
        # With zero wind, the tas and gs will be identical at the ends of the data.
        accel = [0.0, 4.7, 9.5, 14.3, 19.0, 23.8, 28.6, 33.3, 38.1, 42.9, 
                 47.6, 52.4, 57.1, 61.9, 66.7, 71.4, 76.2, 81.0, 85.7, 90.5]
        cas = P('Airspeed', np.ma.array([0]*9+accel[9:]+accel[-1:8:-1]+[0]*9))
        gspd = P('Groundspeed', np.ma.array(accel+accel[::-1]))
        gspd.array[0:3] = [12.0, 12.0, 12.0]
        gspd.array[-4:] = [14.0, 14.0, 14.0, 14.0]
        alt = P('Altitude STD', np.ma.array([0]*40))
        acc = P('Acceleration Forwards', np.ma.array([0.25]*20+[-0.25]*20))
        toffs = buildsection('Takeoff', 0, 18)
        lands = buildsection('Landing', 21, None)
        tas = AirspeedTrue()
        tas.derive(cas, alt, None, toffs, lands, None, gspd, acc)
        expected = accel+accel[::-1]
        ma_test.assert_array_almost_equal(tas.array[3:-4], expected[3:-4], decimal=1)
        ma_test.assert_array_almost_equal(tas.array[:3], [12.0]*3, decimal=1)
        ma_test.assert_array_almost_equal(tas.array[-4:], [14.0]*4, decimal=1)
        # Curiously, the test above only checks the valid samples, so no
        # extrapolation is needed to pass, hence a check on validity is
        # essential !
        self.assertEqual(np.ma.count(tas.array), 40)

    def test_tas_no_gs_extensions(self):
        # With no groundspeed available, the true airspeed is an integration
        # of acceleration from the ends of the available data. The array
        # "speed" corresponds to a 0.25g acceleration and deceleration.
        speed = [0.0, 4.7, 9.5, 14.3, 19.0, 23.8, 28.6, 33.3, 38.1, 42.9, 
                 47.6, 52.4, 57.1, 61.9, 66.7, 71.4, 76.2, 81.0, 85.7, 90.5]
        cas = P('Airspeed', np.ma.array([0]*9+speed[9:]+speed[-1:8:-1]+[0]*9))
        alt = P('Altitude STD', np.ma.array([0]*40))
        acc = P('Acceleration Forwards', np.ma.array([0.25]*20+[-0.25]*20))
        toffs = buildsection('Takeoff', 0, 18)
        lands = buildsection('Landing', 21, None)
        tas = AirspeedTrue()
        tas.derive(cas, alt, None, toffs, lands, None, None, acc)
        expected = speed+speed[::-1]
        ma_test.assert_array_almost_equal(tas.array, expected, decimal=1)
        # Curiously, the test above only checks the valid samples, so no
        # extrapolation is needed to pass, hence a check on validity is
        # essential !
        self.assertEqual(np.ma.count(tas.array), 40)

    def test_tas_rto(self):
        speed = [0.0, 4.7, 9.5, 14.3, 19.0, 23.8, 28.6, 33.3, 38.1, 42.9, 
                 47.6, 52.4, 57.1, 61.9, 66.7, 71.4, 76.2, 81.0, 85.7, 90.5]
        cas = P('Airspeed', np.ma.array([0]*9+speed[9:]+speed[-1:8:-1]+[0]*9))
        alt = P('Altitude STD', np.ma.array([0]*40))
        acc = P('Acceleration Forwards', np.ma.array([0.25]*20+[-0.25]*20))
        rtos = buildsection('Rejected Takeoff', 1, 38)
        tas = AirspeedTrue()
        tas.derive(cas, alt, None, None, None, rtos, None, acc)
        expected = speed+speed[::-1]
        ma_test.assert_array_almost_equal(tas.array, expected, decimal=1)


class TestAltitudeSTDSmoothed(unittest.TestCase):
    def test_can_operate(self):
        opts = AltitudeSTDSmoothed.get_operational_combinations()
        self.assertTrue(('Altitude STD', 'Frame',) in opts)
        self.assertTrue(('Altitude STD (Fine)', 'Altitude STD', 'Frame') in opts)
        self.assertTrue(('Altitude STD (Capt)', 'Altitude STD (FO)', 'Frame') in opts)
    
    def test_derive_atr_42(self):
        frame = Attribute('Frame', 'ATR42_V2_Quad')
        alt = load(os.path.join(test_data_path, 'AltitudeSTDSmoothed_alt.nod'))
        node = AltitudeSTDSmoothed()
        node.derive(None, alt, None, None, frame)
        # np.ma.sum(np.ma.abs(np.ma.diff(alt.array[4800:5000]))) == 2000
        linear_diff = np.ma.sum(np.ma.abs(np.ma.diff(node.array[4800:5000])))
        self.assertTrue(linear_diff < 200)
        max_diff = np.ma.max(np.ma.abs(alt.array - node.array))
        self.assertTrue(max_diff < 100)


class TestAltitudeAAL(unittest.TestCase):
    def test_can_operate(self):
        opts = AltitudeAAL.get_operational_combinations()
        self.assertTrue(('Altitude STD Smoothed', 'Fast') in opts)
        self.assertTrue(('Altitude Radio', 'Altitude STD Smoothed', 'Fast') in opts)
        
    def test_alt_aal_basic(self):
        data = np.ma.array([-3, 0, 30, 80, 250, 560, 220, 70, 20, -5])
        alt_std = P(array=data + 300)
        alt_rad = P(array=data)
        fast_data = np.ma.array([100] * 10)
        phase_fast = Fast()
        phase_fast.derive(Parameter('Airspeed', fast_data))
        alt_aal = AltitudeAAL()
        alt_aal.derive(alt_rad,alt_std, phase_fast)
        expected = np.ma.array([0, 0, 30, 80, 250, 560, 220, 70, 20, 0])
        np.testing.assert_array_equal(expected, alt_aal.array.data)

    def test_alt_aal_bounce_rejection(self):
        data = np.ma.array([-3, 0, 30, 80, 250, 560, 220, 70, 20, -5, 2, 2, 2,
                            -3, -3])
        alt_std = P(array=data + 300)
        alt_rad = P(array=data)
        fast_data = np.ma.array([100] * 15)
        phase_fast = Fast()
        phase_fast.derive(Parameter('Airspeed', fast_data))
        alt_aal = AltitudeAAL()
        alt_aal.derive(alt_rad, alt_std, phase_fast)
        expected = np.ma.array([0, 0, 30, 80, 250, 560, 220, 70, 20, 0, 0, 0, 0,
                                0, 0])
        np.testing.assert_array_equal(expected, alt_aal.array.data)
    
    def test_alt_aal_no_ralt(self):
        data = np.ma.array([-3, 0, 30, 80, 250, 580, 220, 70, 20, 25])
        alt_std = P(array=data + 300)
        slow_and_fast_data = np.ma.array([70] + [85] * 7 + [70]*2)
        phase_fast = Fast()
        phase_fast.derive(Parameter('Airspeed', slow_and_fast_data))
        pitch = P('Pitch', np_ma_ones_like(slow_and_fast_data))
        alt_aal = AltitudeAAL()
        alt_aal.derive(None, alt_std, phase_fast, pitch)
        expected = np.ma.array([0, 0, 30, 80, 250, 560, 200, 50, 0, 0])
        np.testing.assert_array_equal(expected, alt_aal.array.data)

    def test_alt_aal_complex(self):
        testwave = np.ma.cos(np.arange(0, 3.14 * 2 * 5, 0.1)) * -3000 + \
            np.ma.cos(np.arange(0, 3.14 * 2, 0.02)) * -5000 + 7996
        rad_wave = np.copy(testwave)
        rad_wave[110:140] -= 8765 # The ground is 8,765 ft high at this point.
        rad_data = np.ma.masked_greater(rad_wave, 2600)
        phase_fast = buildsection('Fast', 0, len(testwave))
        alt_aal = AltitudeAAL()
        alt_aal.derive(P('Altitude Radio', rad_data),
                       P('Altitude STD', testwave),
                       phase_fast)
        '''
        import matplotlib.pyplot as plt
        plt.plot(testwave)
        plt.plot(rad_data)
        plt.plot(alt_aal.array)
        plt.show()
        '''
        # Check that the waveform reaches the right points.
        np.testing.assert_equal(alt_aal.array[0], 0.0)
        np.testing.assert_almost_equal(alt_aal.array[34], 7013, decimal=0)
        np.testing.assert_almost_equal(alt_aal.array[60], 3308, decimal=0)
        np.testing.assert_almost_equal(alt_aal.array[124], 217, decimal=0)
        np.testing.assert_almost_equal(alt_aal.array[191], 8965, decimal=0)
        np.testing.assert_almost_equal(alt_aal.array[254], 3288, decimal=0)
        np.testing.assert_almost_equal(alt_aal.array[313], 17, decimal=0)

    def test_alt_aal_complex_no_ralt_flying_below_takeoff_airfield(self):
        testwave = np.ma.cos(np.arange(0, 3.14 * 2 * 5, 0.1)) * -2000 + \
            np.ma.cos(np.arange(0, 3.14 * 2, 0.02)) * 5000 + 0
        phase_fast = buildsection('Fast', 0, len(testwave))
        alt_aal = AltitudeAAL()
        alt_aal.derive(None,
                       P('Altitude STD', testwave),
                       phase_fast,
                       P('Pitch', np_ma_ones_like(testwave))
                       )
        '''
        import matplotlib.pyplot as plt
        plt.plot(testwave, '-b')
        plt.plot(alt_aal.array, '-r')
        plt.show()
        '''

    def test_alt_aal_complex_with_mask(self):
        #testwave = np.ma.cos(np.arange(0, 3.14 * 2 * 5, 0.1)) * -3000 + \
            #np.ma.cos(np.arange(0, 3.14 * 2, 0.02)) * -5000 + 7996
        
        # Slope of np.ma.arange(0,5000, 50) reduced to ensure at least one
        # sample point fell in the range 0-100ft for the alt_rad logic to
        # work. DJ.
        std_wave = np.ma.concatenate([np.ma.zeros(50), 
                                      np.ma.arange(0,5000, 50), 
                                      np.ma.zeros(50)+5000, 
                                      np.ma.arange(5000,5500, 200), 
                                      np.ma.zeros(50)+5500, 
                                      np.ma.arange(5500, 0, -500), 
                                      np.ma.zeros(50)])
        rad_wave = np.copy(std_wave) - 8
        rad_data = np.ma.masked_greater(rad_wave, 2600)
        phase_fast = buildsection('Fast', 35, len(std_wave))
        std_wave += 1000
        rad_data[42:48] = np.ma.masked
        alt_aal = AltitudeAAL()
        alt_aal.derive(P('Altitude Radio', np.ma.copy(rad_data)),
                       P('Altitude STD', np.ma.copy(std_wave)),
                       phase_fast,
                       P('Pitch', np_ma_ones_like(rad_data))
                       )
        '''
        import matplotlib.pyplot as plt
        plt.plot(std_wave, '-b')
        plt.plot(rad_data, 'o-r')
        plt.plot(alt_aal.array, '-k')
        plt.show()
        '''
        #  Check alt aal does not try to jump to alt std in masked period of
        #  alt rad
        self.assertEqual(alt_aal.array[45], 0)  # NOT 1000!

    def test_alt_aal_complex_doubled(self):
        testwave = np.ma.cos(np.arange(0, 3.14 * 2, 0.02)) * -5000 + 5500
        rad_wave = np.copy(testwave)-500
        #rad_wave[110:140] -= 8765 # The ground is 8,765 ft high at this point.
        rad_data = np.ma.masked_greater(rad_wave, 2600)
        double_test = np.ma.concatenate((testwave, testwave))
        double_rad = np.ma.concatenate((rad_data, rad_data))
        phase_fast = buildsection('Fast', 0, 2*len(testwave))
        alt_aal = AltitudeAAL()
        alt_aal.derive(P('Altitude Radio', double_rad),
                       P('Altitude STD', double_test),
                       phase_fast)
        '''
        import matplotlib.pyplot as plt
        plt.plot(double_test, '-b')
        plt.plot(double_rad, 'o-r')
        plt.plot(alt_aal.array, '-k')
        plt.show()
        '''
        self.assertNotEqual(alt_aal.array[200], 0.0)
        np.testing.assert_equal(alt_aal.array[0], 0.0)

    def test_alt_aal_complex_doubled_with_touch_and_go(self):
        testwave = np.ma.cos(np.arange(0, 3.14 * 2, 0.02)) * -5000 + 5000
        rad_wave = np.copy(testwave)-500
        #rad_wave[110:140] -= 8765 # The ground is 8,765 ft high at this point.
        rad_data = np.ma.masked_greater(rad_wave, 2600)
        double_test = np.ma.concatenate((testwave, testwave))
        double_rad = np.ma.concatenate((rad_data, rad_data))
        phase_fast = buildsection('Fast', 0, 2*len(testwave))
        alt_aal = AltitudeAAL()
        alt_aal.derive(P('Altitude Radio', double_rad),
                       P('Altitude STD', double_test),
                       phase_fast)
        '''
        import matplotlib.pyplot as plt
        plt.plot(double_test, '-b')
        plt.plot(double_rad, 'o-r')
        plt.plot(alt_aal.array, '-k')
        plt.show()
        '''
        np.testing.assert_equal(alt_aal.array[300:310], [0.0]*10)
        

    def test_alt_aal_complex_no_rad_alt(self):
        testwave = np.ma.cos(np.arange(0, 3.14 * 2 * 5, 0.1)) * -3000 + \
            np.ma.cos(np.arange(0, 3.14 * 2, 0.02)) * -5000 + 7996
        testwave[255:]=testwave[254]
        testwave[:5]=500.0
        phase_fast = buildsection('Fast', 0, 254)
        alt_aal = AltitudeAAL()
        alt_aal.derive(None, 
                       P('Altitude STD', testwave),
                       phase_fast,
                       P('Pitch', np_ma_ones_like(testwave))
                       )
        '''
        import matplotlib.pyplot as plt
        plt.plot(testwave)
        plt.plot(alt_aal.array)
        plt.show()
        '''
        np.testing.assert_equal(alt_aal.array[0], 0.0)
        np.testing.assert_almost_equal(alt_aal.array[34], 6620, decimal=0)
        np.testing.assert_almost_equal(alt_aal.array[60], 2915, decimal=0)
        np.testing.assert_almost_equal(alt_aal.array[124], 8594, decimal=0)
        np.testing.assert_almost_equal(alt_aal.array[191], 8594, decimal=0)
        np.testing.assert_almost_equal(alt_aal.array[254], 0, decimal=0)

    def test_alt_aal_no_rad_alt_pitch_inclusion(self):
        testwave = np.ma.cos(np.arange(0, 3.14 * 2.7, 0.3)) * -280 + 380
        join = 20
        x = testwave[join]
        n = len(testwave) - join
        testwave[join:] = np.linspace(x, x-80, n)
        phase_fast = buildsection('Fast', 0, 28)
        pch = np_ma_ones_like(testwave)
        pch[23:27]=2.0
        alt_aal = AltitudeAAL()
        alt_aal.derive(None, 
                       P('Altitude STD', testwave),
                       phase_fast,
                       P('Pitch', pch)
                       )
        
        '''
        import matplotlib.pyplot as plt
        plt.plot(testwave)
        plt.plot(alt_aal.array)
        plt.show()
        '''
        # Check that AAL goes to zero at the last pitch high point and stays there.
        self.assertGreater(alt_aal.array[25], 0.0)
        self.assertEqual(alt_aal.array[26], 0.0)
        self.assertEqual(alt_aal.array[27], 0.0)

    def test_alt_aal_misleading_rad_alt(self):
        # Spurious Altitude Radio data when Altitude STD was above 30000 ft was
        # causing Altitude AAL to be shifted down to -30000 ft.
        alt_std = load(os.path.join(
            test_data_path, 'AltitudeAAL_AltitudeSTDSmoothed.nod'))
        alt_rad = load(os.path.join(
            test_data_path, 'AltitudeAAL_AltitudeRadio.nod'))
        fast = load(os.path.join(
            test_data_path, 'AltitudeAAL_Fast.nod'))
        alt_aal = AltitudeAAL()
        alt_aal.derive(alt_rad, alt_std, fast)
        self.assertEqual(np.ma.min(alt_aal.array), 0.0)
        
    @unittest.skip('Test Not Implemented')
    def test_alt_aal_faulty_alt_rad(self):
        '''
        When 'Altitude Radio' does not reach 0 after touchdown due to an arinc
        signal being recorded, 'Altitude AAL' did not fill the second half of
        its array. Since the array is initialised as zeroes
        '''
        hdf_copy = copy_file(os.path.join(test_data_path,
                                          'alt_aal_faulty_alt_rad.hdf5'),
                             postfix='_test_copy')
        process_flight(hdf_copy, 'G-DEMA', {
            'Engine Count': 2,
            'Frame': '737-3C', # TODO: Change.
            'Manufacturer': 'Boeing',
            'Model': 'B737-86N',
            'Precise Positioning': True,
            'Series': 'B767-300',
        })
        with hdf_file(hdf_copy) as hdf:
            hdf['Altitude AAL']
            self.assertTrue(False, msg='Test not implemented.')
    
    @unittest.skip('Test Not Implemented')
    def test_alt_aal_without_alt_rad(self):
        '''
        When 'Altitude Radio' is not available, 'Altitude AAL' is created from
        'Altitude STD' using the cycle_finder and peak_curvature algorithms.
        Currently, cycle_finder is accurately locating the index where the
        aircraft begins to climb. This section of data is passed into 
        peak_curvature, which is designed to find the first curve in a piece of
        data. The problem is that data from before the first curve, where the 
        aircraft starts climbing, is not included, and peak_curvature detects
        the second curve at approximately 120 feet.
        '''
        hdf_copy = copy_file(os.path.join(test_data_path,
                                          'alt_aal_without_alt_rad.hdf5'),
                             postfix='_test_copy')
        process_flight(hdf_copy, 'G-DEMA', {
            'Engine Count': 2,
            'Frame': '737-3C', # TODO: Change.
            'Manufacturer': 'Boeing',
            'Model': 'B737-86N',
            'Precise Positioning': True,
            'Series': 'B767-300',
        })
        with hdf_file(hdf_copy) as hdf:
            hdf['Altitude AAL']
            self.assertTrue(False, msg='Test not implemented.')

    @unittest.skip('Test Not Implemented')
    def test_alt_aal_training_flight(self):
        alt_std = load(os.path.join(test_data_path,
                                    'TestAltitudeAAL-training-alt_std.nod'))
        alt_rad = load(os.path.join(test_data_path,
                                    'TestAltitudeAAL-training-alt_rad.nod'))
        fasts = load(os.path.join(test_data_path,
                                    'TestAltitudeAAL-training-fast.nod'))
        alt_aal = AltitudeAAL()
        alt_aal.derive(alt_rad, alt_std, fasts)
        peak_detect = np.ma.masked_where(alt_aal.array < 500, alt_aal.array)
        peaks = np.ma.clump_unmasked(peak_detect)
        # Check to test that all 6 altitude sections are inculded in alt aal
        self.assertEqual(len(peaks), 6)

    @unittest.skip('Test Not Implemented')
    def test_alt_aal_goaround_flight(self):
        alt_std = load(os.path.join(test_data_path,
                                    'TestAltitudeAAL-goaround-alt_std.nod'))
        alt_rad = load(os.path.join(test_data_path,
                                    'TestAltitudeAAL-goaround-alt_rad.nod'))
        fasts = load(os.path.join(test_data_path,
                                    'TestAltitudeAAL-goaround-fast.nod'))
        alt_aal = AltitudeAAL()
        alt_aal.derive(alt_rad, alt_std, fasts)
        difs = np.diff(alt_aal.array)
        index, value = max_value(np.abs(difs))
        # Check to test that the step occurs during cruse and not the go-around
        self.assertTrue(index in range(1290, 1850))
        
    def test_find_liftoff_start_on_herc(self):
        # Herc (L100) climbs in a straight line without noticable concave
        # curvature at liftoff; ensure index is kept close
        aal = AltitudeAAL(frequency=2)
        herc_alt_std = np.ma.array([
       -143.20034375,    0.        , -142.46179687,    0.        ,
       -140.98470312,    0.        , -140.24615625,    0.        ,
       -138.7690625 ,    0.        , -138.03051562,    0.        ,
       -135.814875  ,    0.        , -134.33778125,    0.        ,
       -132.8606875 ,    0.        , -132.8606875 ,    0.        ,
       -130.64504687,    0.        , -129.9065    ,    0.        ,
       -128.42940625,    0.        , -127.69085937,    0.        ,
       -125.47521875,    0.        , -123.25957812,    0.        ,
       -121.0439375 ,    0.        , -118.82829687,    0.        ,
       -116.61265625,    0.        , -114.39701562,    0.        ,
       -110.70428125,    0.        , -108.48864062,    0.        ,
       -106.273     ,    0.        , -102.58026562,    0.        ,
        -99.62607812,    0.        ,  -97.4104375 ,    0.        ,
        -92.97915625,    0.        ,  -89.28642187,    0.        ,
        -84.85514062,    0.        ,  -81.16240625,    0.        ,
        -78.20821875,    0.        , -901.50908125, -881.86373437,
       -862.2183875 , -838.95416094, -815.68993437, -792.05643437,
       -768.42293437, -749.14686094, -729.8707875 , -706.60656094,
       -683.34233437, -664.06626094, -644.7901875 , -617.16853437,
       -589.54688125, -565.54410781, -541.54133437, -526.6226875 ,
       -511.70404062, -488.8090875 , -465.91413437, -450.9954875 ,
       -436.07684062, -421.89674062, -407.71664062, -389.17911406,
       -370.6415875 , -347.37736094, -324.11313437, -304.83706094,
       -285.5609875 , -262.29676094, -239.03253437, -224.1138875 ,
       -209.19524062, -186.3002875 , -163.40533437, -144.12926094,
       -124.8531875 , -110.30381406,  -95.75444062,  -81.57434062,
        -67.39424062,  -53.21414062,  -39.03404062,  -24.85394062,
        -10.67384062,    3.50625938,   17.68635938,   27.50903281,
         37.33170625,   51.14253281,   64.95335938,   79.13345938,
         93.31355938,  107.49365938,  121.67375938,  127.13900625,
        132.60425313,  150.40323281,  168.2022125 ,  182.75158594,
        197.30095938,  211.48105938,  225.66115938,  239.84125938,
        254.02135938,  268.20145938,  282.38155938,  296.56165938,
        310.74175938,  324.92185938,  339.10195938,  353.28205938,
        367.46215938,  381.64225938,  395.82235938,  414.35988594,
        432.8974125 ,  451.8042125 ,  470.7110125 ,  485.26038594,
        499.80975938,  513.98985938,  528.16995938,  546.70748594,
        565.2450125 ,  579.79438594,  594.34375938,  608.52385938,
        622.70395938,  645.5989125 ,  668.49386563,  687.76993906,
        707.0460125 ,  721.59538594,  736.14475938,  759.0397125 ])
        herc_alt_std[:62] = np.ma.masked
        idx = aal.find_liftoff_start(herc_alt_std)
        self.assertEqual(idx, 63)

class TestAimingPointRange(unittest.TestCase):
    def test_basic_scaling(self):
        approaches = App(items=[ApproachItem(
            'Landing', slice(3, 8),
            runway={'end': 
                    {'elevation': 3294,
                     'latitude': 31.497511,
                     'longitude': 65.833933},
                    'start': 
                    {'elevation': 3320,
                     'latitude': 31.513997,
                     'longitude': 65.861714}})])
        app_rng=P('Approach Range',
                  array=np.ma.arange(10000.0, -2000.0, -1000.0))
        apr = AimingPointRange()
        apr.derive(app_rng, approaches)
        # convoluted way to check masked outside slice !
        self.assertEqual(apr.array[0].mask, np.ma.masked.mask)
        self.assertAlmostEqual(apr.array[4], 1.67, places=2)
        
        
class TestAltitudeAALForFlightPhases(unittest.TestCase):
    def test_can_operate(self):
        expected = [('Altitude AAL',)]
        opts = AltitudeAALForFlightPhases.get_operational_combinations()
        self.assertEqual(opts, expected)
        
    def test_altitude_AAL_for_flight_phases_basic(self):
        alt_4_ph = AltitudeAALForFlightPhases()
        alt_4_ph.derive(Parameter('Altitude AAL', 
                                  np.ma.array(data=[0,100,200,100,0],
                                              mask=[0,0,1,1,0])))
        expected = np.ma.array(data=[0,100,66,33,0],mask=False)
        # ...because data interpolates across the masked values and integer
        # values are rounded.
        ma_test.assert_array_equal(alt_4_ph.array, expected)



'''
class TestAltitudeForFlightPhases(unittest.TestCase):
    def test_can_operate(self):
        expected = [('Altitude STD',)]
        opts = AltitudeForFlightPhases.get_operational_combinations()
        self.assertEqual(opts, expected)

    def test_altitude_for_phases_repair(self):
        alt_4_ph = AltitudeForFlightPhases()
        raw_data = np.ma.array([0,1,2])
        raw_data[1] = np.ma.masked
        alt_4_ph.derive(Parameter('Altitude STD', raw_data, 1,0.0))
        expected = np.ma.array([0,0,0],mask=False)
        np.testing.assert_array_equal(alt_4_ph.array, expected)
        
    def test_altitude_for_phases_hysteresis(self):
        alt_4_ph = AltitudeForFlightPhases()
        testwave = np.sin(np.arange(0,6,0.1))*200
        alt_4_ph.derive(Parameter('Altitude STD', np.ma.array(testwave), 1,0.0))
        answer = np.ma.array(data=[50.0]*3+
                             list(testwave[3:6])+
                             [np.ma.max(testwave)-100.0]*21+
                             list(testwave[27:39])+
                             [testwave[-1]-50.0]*21,
                             mask = False)
        np.testing.assert_array_almost_equal(alt_4_ph.array, answer)
        '''


class TestAltitudeQNH(unittest.TestCase, NodeTest):
    def setUp(self):
        self.node_class = AltitudeQNH
        self.operational_combinations = [
            ('Altitude AAL', 'Altitude Peak'),
            ('Altitude AAL', 'Altitude Peak', 'FDR Landing Airport'),
            ('Altitude AAL', 'Altitude Peak', 'FDR Landing Runway'),
            ('Altitude AAL', 'Altitude Peak', 'FDR Takeoff Airport'),
            ('Altitude AAL', 'Altitude Peak', 'FDR Takeoff Runway'),
            ('Altitude AAL', 'Altitude Peak', 'FDR Landing Airport', 'FDR Landing Runway'),
            ('Altitude AAL', 'Altitude Peak', 'FDR Landing Airport', 'FDR Takeoff Airport'),
            ('Altitude AAL', 'Altitude Peak', 'FDR Landing Airport', 'FDR Takeoff Runway'),
            ('Altitude AAL', 'Altitude Peak', 'FDR Landing Runway', 'FDR Takeoff Airport'),
            ('Altitude AAL', 'Altitude Peak', 'FDR Landing Runway', 'FDR Takeoff Runway'),
            ('Altitude AAL', 'Altitude Peak', 'FDR Takeoff Airport', 'FDR Takeoff Runway'),
            ('Altitude AAL', 'Altitude Peak', 'FDR Landing Airport', 'FDR Landing Runway', 'FDR Takeoff Airport'),
            ('Altitude AAL', 'Altitude Peak', 'FDR Landing Airport', 'FDR Landing Runway', 'FDR Takeoff Runway'),
            ('Altitude AAL', 'Altitude Peak', 'FDR Landing Airport', 'FDR Takeoff Airport', 'FDR Takeoff Runway'),
            ('Altitude AAL', 'Altitude Peak', 'FDR Landing Runway', 'FDR Takeoff Airport', 'FDR Takeoff Runway'),
            ('Altitude AAL', 'Altitude Peak', 'FDR Landing Airport', 'FDR Landing Runway', 'FDR Takeoff Airport', 'FDR Takeoff Runway'),
        ]
        data = [np.ma.arange(0, 1000, step=30)]
        data.append(data[0][::-1] + 50)
        self.alt_aal = P(name='Altitude AAL', array=np.ma.concatenate(data))
        self.alt_peak = KTI(name='Altitude Peak', items=[KeyTimeInstance(name='Altitude Peak', index=len(self.alt_aal.array) / 2)])
        self.land_fdr_apt = A(name='FDR Landing Airport', value={'id': 10, 'elevation': 100})
        self.land_fdr_rwy = A(name='FDR Landing Runway', value={'ident': '27L', 'start': {'elevation': 90}, 'end': {'elevation': 110}})
        self.toff_fdr_apt = A(name='FDR Takeoff Airport', value={'id': 20, 'elevation': 50})
        self.toff_fdr_rwy = A(name='FDR Takeoff Runway', value={'ident': '09R', 'start': {'elevation': 40}, 'end': {'elevation': 60}})

        self.expected = []
        peak = self.alt_peak[0].index

        # Ensure that we have a sensible drop at the splitting point...
        self.alt_aal.array[peak + 1] += 30
        self.alt_aal.array[peak] -= 30

        # 1. Data same as Altitude AAL, no mask applied:
        data = np.ma.copy(self.alt_aal.array)
        self.expected.append(data)
        # 2. None masked, data Altitude AAL, +50 ft t/o, +100 ft ldg:
        data = np.ma.array([50, 80, 110, 140, 170, 200, 230, 260, 290, 320,
            350, 351, 352, 354, 355, 357, 358, 360, 361, 363, 364, 366, 367,
            368, 370, 371, 373, 374, 376, 377, 379, 380, 382, 383, 385, 386,
            387, 389, 390, 392, 393, 395, 396, 398, 399, 401, 402, 403, 405,
            406, 408, 409, 411, 412, 414, 415, 417, 418, 420, 390, 360, 330,
            300, 270, 240, 210, 180, 150])
        data.mask = False
        self.expected.append(data)
        # 3. Data Altitude AAL, +50 ft t/o; ldg assumes t/o elevation:
        data = np.ma.copy(self.alt_aal.array)
        data += 50
        self.expected.append(data)
        # 4. Data Altitude AAL, +100 ft ldg; t/o assumes ldg elevation:
        data = np.ma.copy(self.alt_aal.array)
        data += 100
        self.expected.append(data)

    def test_derive__function_calls(self):
        alt_qnh = self.node_class()
        alt_qnh._calc_apt_elev = Mock(return_value=0)
        alt_qnh._calc_rwy_elev = Mock(return_value=0)
        # Check no airport/runway information results in a fully masked copy of Altitude AAL:
        alt_qnh.derive(self.alt_aal, self.alt_peak)
        self.assertFalse(alt_qnh._calc_apt_elev.called, 'method should not have been called')
        self.assertFalse(alt_qnh._calc_rwy_elev.called, 'method should not have been called')
        alt_qnh._calc_apt_elev.reset_mock()
        alt_qnh._calc_rwy_elev.reset_mock()
        # Check everything works calling with runway details:
        alt_qnh.derive(self.alt_aal, self.alt_peak, None, self.land_fdr_rwy, None, self.toff_fdr_rwy)
        self.assertFalse(alt_qnh._calc_apt_elev.called, 'method should not have been called')
        alt_qnh._calc_rwy_elev.assert_has_calls([
            call(self.toff_fdr_rwy.value),
            call(self.land_fdr_rwy.value),
        ])
        alt_qnh._calc_apt_elev.reset_mock()
        alt_qnh._calc_rwy_elev.reset_mock()
        # Check everything works calling with airport details:
        alt_qnh.derive(self.alt_aal, self.alt_peak, self.land_fdr_apt, None, self.toff_fdr_apt, None)
        alt_qnh._calc_apt_elev.assert_has_calls([
            call(self.toff_fdr_apt.value),
            call(self.land_fdr_apt.value),
        ])
        self.assertFalse(alt_qnh._calc_rwy_elev.called, 'method should not have been called')
        alt_qnh._calc_apt_elev.reset_mock()
        alt_qnh._calc_rwy_elev.reset_mock()
        # Check everything works calling with runway and airport details:
        alt_qnh.derive(self.alt_aal, self.alt_peak, self.land_fdr_apt, self.land_fdr_rwy, self.toff_fdr_apt, self.toff_fdr_rwy)
        self.assertFalse(alt_qnh._calc_apt_elev.called, 'method should not have been called')
        alt_qnh._calc_rwy_elev.assert_has_calls([
            call(self.toff_fdr_rwy.value),
            call(self.land_fdr_rwy.value),
        ])
        alt_qnh._calc_apt_elev.reset_mock()
        alt_qnh._calc_rwy_elev.reset_mock()

    def test_derive__output(self):
        alt_qnh = self.node_class()
        # Check no airport/runway information results in a fully masked copy of Altitude AAL:
        alt_qnh.derive(self.alt_aal, self.alt_peak)
        ma_test.assert_masked_array_approx_equal(alt_qnh.array, self.expected[0])
        self.assertEqual(alt_qnh.offset, self.alt_aal.offset)
        self.assertEqual(alt_qnh.frequency, self.alt_aal.frequency)
        # Check everything works calling with runway details:
        alt_qnh.derive(self.alt_aal, self.alt_peak, None, self.land_fdr_rwy, None, self.toff_fdr_rwy)
        ma_test.assert_masked_array_approx_equal(alt_qnh.array, self.expected[1])
        self.assertEqual(alt_qnh.offset, self.alt_aal.offset)
        self.assertEqual(alt_qnh.frequency, self.alt_aal.frequency)
        # Check everything works calling with airport details:
        alt_qnh.derive(self.alt_aal, self.alt_peak, self.land_fdr_apt, None, self.toff_fdr_apt, None)
        ma_test.assert_masked_array_approx_equal(alt_qnh.array, self.expected[1])
        self.assertEqual(alt_qnh.offset, self.alt_aal.offset)
        self.assertEqual(alt_qnh.frequency, self.alt_aal.frequency)
        # Check everything works calling with runway and airport details:
        alt_qnh.derive(self.alt_aal, self.alt_peak, self.land_fdr_apt, self.land_fdr_rwy, self.toff_fdr_apt, self.toff_fdr_rwy)
        ma_test.assert_masked_array_approx_equal(alt_qnh.array, self.expected[1])
        self.assertEqual(alt_qnh.offset, self.alt_aal.offset)
        self.assertEqual(alt_qnh.frequency, self.alt_aal.frequency)
        # Check second half masked when no elevation at landing:
        alt_qnh.derive(self.alt_aal, self.alt_peak, None, None, self.toff_fdr_apt, self.toff_fdr_rwy)
        ma_test.assert_masked_array_approx_equal(alt_qnh.array, self.expected[2])
        self.assertEqual(alt_qnh.offset, self.alt_aal.offset)
        self.assertEqual(alt_qnh.frequency, self.alt_aal.frequency)
        # Check first half masked when no elevation at takeoff:
        alt_qnh.derive(self.alt_aal, self.alt_peak, self.land_fdr_apt, self.land_fdr_rwy, None, None)
        ma_test.assert_masked_array_approx_equal(alt_qnh.array, self.expected[3])
        self.assertEqual(alt_qnh.offset, self.alt_aal.offset)
        self.assertEqual(alt_qnh.frequency, self.alt_aal.frequency)
    
    def test_new_version(self):
        xalt_aal=P('Altitude AAL', np.ma.array([0]*5+range(0,15000,1000)+[10000]*4+range(10000,-1000,-1000)+[0]*5))
        xalt_std=P('Altitude STD', np.ma.array([1000]*5+range(1000,16000,1000)+[15000]*4+range(15000,4000,-1000)+[4000]*5))
        xtocs=KTI('Top Of Climb', 19)
        xtods=KTI('Top Of Descent', 24)
        xclimbs=buildsection('Climb', 7, 19)
        xdescents=buildsection('Descent', 24, 34)
        alt_qnh = self.node_class()
        alt_qnh.derive(xalt_aal, xalt_std, self.alt_peak, 
                       self.land_fdr_apt, self.land_fdr_rwy, 
                       self.toff_fdr_apt, self.toff_fdr_rwy,
                       xclimbs, xdescents)
        self.assertEqual(alt_qnh.array[2], 50.0) # Takeoff elevation
        self.assertEqual(alt_qnh.array[36], 100.0) # Landing elevation
        self.assertEqual(alt_qnh.array[22],15000.0) # Cruise at STD
        
    def test_trap_alt_difference(self):
        xalt_aal=P('Altitude AAL', np.ma.array([0]*5+range(0,15000,1000)+[10000]*4+range(10000,-1000,-1000)+[0]*5))
        xalt_std=P('Altitude STD', np.ma.array([1000]*5+range(1000,16000,1000)+[15000]*4+range(15000,4000,-1000)+[4000]*5))
        xtocs=KTI('Top Of Climb', 19)
        xtods=KTI('Top Of Descent', 24)
        xclimbs=buildsection('Climb', 7, 19)
        xdescents=buildsection('Descent', 24, 32)
        alt_qnh = self.node_class()
        self.assertRaises(ValueError, alt_qnh.derive,
                          xalt_aal, xalt_std, self.alt_peak, self.land_fdr_apt, 
                          self.land_fdr_rwy, self.toff_fdr_apt, 
                          self.toff_fdr_rwy, xclimbs, xdescents)
        

class TestAltitudeRadio(unittest.TestCase):
    """
    def test_can_operate(self):
        expected = [('Altitude Radio Sensor', 'Pitch',
                     'Main Gear To Altitude Radio')]
        opts = AltitudeRadio.get_operational_combinations()
        self.assertEqual(opts, expected)
    """
    
    def test_altitude_radio_737_3C(self):
        alt_rad = AltitudeRadio()
        alt_rad.derive(Parameter('Altitude Radio (A)', 
                                 np.ma.array([10.0,10.0,10.0,10.0,10.1]*2), 0.5,  0.0),
                       Parameter('Altitude Radio (B)',
                                 np.ma.array([20.0,20.0,20.0,20.0,20.2]), 0.25, 1.0),
                       Parameter('Altitude Radio (C)',
                                 np.ma.array([30.0,30.0,30.0,30.0,30.3]), 0.25, 3.0),
                       None, None, None, None, None)
        answer = np.ma.array(data=[17.5]*80, mask=[True] + (79 * [False]))
        ma_test.assert_array_almost_equal(alt_rad.array, answer, decimal=0)
        self.assertEqual(alt_rad.offset, 0.0)
        self.assertEqual(alt_rad.frequency, 4.0)

    def test_altitude_radio_737_5_EFIS(self):
        alt_rad = AltitudeRadio()
        alt_rad.derive(Parameter('Altitude Radio (A)', 
                                 np.ma.array([10.0,10.0,10.0,10.0,10.1]), 0.5, 0.0),
                       Parameter('Altitude Radio (B)',
                                 np.ma.array([20.0,20.0,20.0,20.0,20.2]), 0.5, 1.0),
                       None, None, None, None, None, None)
        answer = np.ma.array(data=[15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.1, 15.1, 15.1, 15.1, 15.2, 15.2, 15.3, 15.3, 15.4],
                             mask=[True] + ([False] * 38) + [True])
        ma_test.assert_array_almost_equal(alt_rad.array, answer, decimal=1)
        self.assertEqual(alt_rad.offset, 0.0)
        self.assertEqual(alt_rad.frequency, 4.0)

    def test_altitude_radio_737_5_Analogue(self):
        alt_rad = AltitudeRadio()
        alt_rad.derive(Parameter('Altitude Radio (A)', 
                                 np.ma.array([10.0,10.0,10.0,10.0,10.1]), 0.5, 0.0),
                       Parameter('Altitude Radio (B)',
                                 np.ma.array([20.0,20.0,20.0,20.0,20.2]), 0.5, 1.0),
                       None, None, None, None, None, None)
        answer = np.ma.array(data=[
            15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0,
            15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0,
            15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.1, 15.1,
            15.1, 15.1, 15.2, 15.2, 15.3, 15.3, 15.4], mask=[True] + (38 * [False]) + [False])
        ma_test.assert_array_almost_equal(alt_rad.array, answer, decimal=1)
        self.assertEqual(alt_rad.offset, 0.0)
        self.assertEqual(alt_rad.frequency, 4.0)
    
    def test_altitude_radio_787(self):
        alt_rad = AltitudeRadio()
        alt_rad.derive(None, None, None,
                       Parameter('Altitude Radio (L)', 
                                 np.ma.array([10.0,10.0,10.0,10.0,10.1]), 0.5, 0.0),
                       Parameter('Altitude Radio (R)',
                                 np.ma.array([20.0,20.0,20.0,20.0,20.2]), 0.5, 1.0),
                       None, None, None)
        answer = np.ma.array(data=[15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.1, 15.1, 15.1, 15.1, 15.2, 15.2, 15.3, 15.3, 15.4],
                             mask=[True] + (38 * [False]) + [False])
        ma_test.assert_array_almost_equal(alt_rad.array, answer, decimal=1)
        self.assertEqual(alt_rad.offset, 0.0)
        self.assertEqual(alt_rad.frequency, 4.0)
        
    def test_altitude_radio_A320(self):
        # strictly these are two flights, but that should not matter
        fast = S(frequency=0.5,
                 items=[Section('Fast', slice(336, 5397), 336, 5397),
                        Section('Fast', slice(5859, 11520), 5859, 11520)])
        radioA = load(os.path.join(
            test_data_path, 'A320_Altitude_Radio_A_overflow.nod'))
        radioB = load(os.path.join(
            test_data_path, 'A320_Altitude_Radio_B_overflow.nod'))
        
        rad = AltitudeRadio()
        rad.derive(radioA, radioB, None, None, None, None, None, None, None, 
                   fast=fast, family=A('Family', 'A320'))
        
        sects = np.ma.clump_unmasked(rad.array)
        self.assertEqual(len(sects), 4)
        for sect in sects[0::2]:
            # takeoffs
            self.assertAlmostEqual(rad.array[sect.start] / 10., 0, 0)
        for sect in sects[1::2]:
            # landings
            self.assertAlmostEqual(rad.array[sect.stop - 1] / 10., 0, 0)
    
    def test_altitude_radio_A330(self):
        fast = buildsection('Fast', 480, 31032)
        radioA = load(os.path.join(
            test_data_path, 'A330_AltitudeRadio_A_overflow_8191.nod'))
        radioB = load(os.path.join(
            test_data_path, 'A330_AltitudeRadio_B_overflow_8191.nod'))
        
        rad = AltitudeRadio()
        rad.derive(radioA, radioB, None, None, None, None, None, None, None, 
                   fast=fast, family=A('Family', 'A330'))
        
        sects = np.ma.clump_unmasked(rad.array)
        self.assertEqual(len(sects), 4)
        self.assertEqual(sects[0].start, 17)
        self.assertEqual(sects[0].stop, 2763)
        self.assertAlmostEqual(rad.array[2762], 5524, places=0)
        self.assertEqual(sects[1].start, 122453)
        self.assertAlmostEqual(rad.array[122453], 5455, places=0)


    def test_altitude_radio_CL_600(self):
        alt_rad = AltitudeRadio()
        fast = buildsection('Fast', 0, 6)
        alt_rad.derive(None, None, None,
                       Parameter('Altitude Radio (L)', 
                                 np.ma.array(range(5,-5,-1)+range(-5,15)), 1.0, 0.0),
                       None, None, None, None,
                       Parameter('Pitch',
                                 np.ma.array([0.0]*30+[5.0]*30+[10.0]*30+[20.0]*30), 4.0, 0.0),
                       fast=fast, 
                       family=A('Family', 'CL-600'))
        self.assertAlmostEqual(alt_rad.array.data[4], 2.5) # -1.5ft offset
        self.assertEqual(alt_rad.array.data[36], -3.675) # -1.5ft & 5deg
        self.assertEqual(alt_rad.array.data[76], 6.15) # -1.5ft & 10 deg
        
'''
class TestAltitudeRadioForFlightPhases(unittest.TestCase):
    def test_can_operate(self):
        expected = [('Altitude Radio',)]
        opts = AltitudeRadioForFlightPhases.get_operational_combinations()
        self.assertEqual(opts, expected)

    def test_altitude_for_radio_phases_repair(self):
        alt_4_ph = AltitudeRadioForFlightPhases()
        raw_data = np.ma.array([0,1,2])
        raw_data[1] = np.ma.masked
        alt_4_ph.derive(Parameter('Altitude Radio', raw_data, 1,0.0))
        expected = np.ma.array([0,0,0],mask=False)
        np.testing.assert_array_equal(alt_4_ph.array, expected)
        '''


"""
class TestAltitudeQNH(unittest.TestCase):
    # Needs airport database entries simulated. TODO.

"""    
    
'''
class TestAltitudeSTD(unittest.TestCase):
    def test_can_operate(self):
        self.assertEqual(AltitudeSTD.get_operational_combinations(),
          [('Altitude STD (Coarse)', 'Altitude STD (Fine)'),
           ('Altitude STD (Coarse)', 'Vertical Speed')])
    
    def test__high_and_low(self):
        high_values = np.ma.array([15000, 16000, 17000, 18000, 19000, 20000,
                                   19000, 18000, 17000, 16000],
                                  mask=[False] * 9 + [True])
        low_values = np.ma.array([15500, 16500, 17500, 17800, 17800, 17800,
                                  17800, 17800, 17500, 16500],
                                 mask=[False] * 8 + [True] + [False])
        alt_std_high = Parameter('Altitude STD High', high_values)
        alt_std_low = Parameter('Altitude STD Low', low_values)
        alt_std = AltitudeSTD()
        result = alt_std._high_and_low(alt_std_high, alt_std_low)
        ma_test.assert_equal(result,
                             np.ma.masked_array([15500, 16500, 17375, 17980, 19000,
                                                 20000, 19000, 17980, 17375, 16500],
                                                mask=[False] * 8 + 2 * [True]))
    
    @patch('analysis_engine.derived_parameters.first_order_lag')
    def test__rough_and_ivv(self, first_order_lag):
        alt_std = AltitudeSTD()
        alt_std_rough = Parameter('Altitude STD Rough',
                                  np.ma.array([60, 61, 62, 63, 64, 65],
                                              mask=[False] * 5 + [True]))
        first_order_lag.side_effect = lambda arg1, arg2, arg3: arg1
        ivv = Parameter('Inertial Vertical Speed',
                        np.ma.array([60, 120, 180, 240, 300, 360],
                                    mask=[False] * 4 + [True] + [False]))
        result = alt_std._rough_and_ivv(alt_std_rough, ivv)
        ma_test.assert_equal(result,
                             np.ma.masked_array([61, 63, 65, 67, 0, 0],
                                                mask=[False] * 4 + [True] * 2))
    
    def test_derive(self):
        alt_std = AltitudeSTD()
        # alt_std_high and alt_std_low passed in.
        alt_std._high_and_low = Mock()
        high_and_low_array = 3
        alt_std._high_and_low.return_value = high_and_low_array
        alt_std_high = 1
        alt_std_low = 2
        alt_std.derive(alt_std_high, alt_std_low, None, None)
        alt_std._high_and_low.assert_called_once_with(alt_std_high, alt_std_low)
        self.assertEqual(alt_std.array, high_and_low_array)
        # alt_std_rough and ivv passed in.
        rough_and_ivv_array = 6
        alt_std._rough_and_ivv = Mock()
        alt_std._rough_and_ivv.return_value = rough_and_ivv_array
        alt_std_rough = 4        
        ivv = 5
        alt_std.derive(None, None, alt_std_rough, ivv)
        alt_std._rough_and_ivv.assert_called_once_with(alt_std_rough, ivv)
        self.assertEqual(alt_std.array, rough_and_ivv_array)
        # All parameters passed in (improbable).
        alt_std.derive(alt_std_high, alt_std_low, alt_std_rough, ivv)
        self.assertEqual(alt_std.array, high_and_low_array)
        '''


class TestAltitudeTail(unittest.TestCase):
    def test_can_operate(self):
        expected = [('Altitude Radio', 'Pitch',
                     'Ground To Lowest Point Of Tail',
                     'Main Gear To Lowest Point Of Tail')]
        opts = AltitudeTail.get_operational_combinations()
        self.assertEqual(opts, expected)
        
    def test_altitude_tail(self):
        talt = AltitudeTail()
        talt.derive(Parameter('Altitude Radio', np.ma.zeros(10), 1,0.0),
                    Parameter('Pitch', np.ma.array(range(10))*2, 1,0.0),
                    Attribute('Ground To Lowest Point Of Tail', 10.0/METRES_TO_FEET),
                    Attribute('Main Gear To Lowest Point Of Tail', 35.0/METRES_TO_FEET))
        result = talt.array
        # At 35ft and 18deg nose up, the tail just scrapes the runway with 10ft
        # clearance at the mainwheels...
        answer = np.ma.array(data=[10.0,
                                   8.77851761541,
                                   7.55852341896,
                                   6.34150378563,
                                   5.1289414664,
                                   3.92231378166,
                                   2.72309082138,
                                   1.53273365401,
                                   0.352692546405,
                                   -0.815594803123],
                             dtype=np.float, mask=False)
        np.testing.assert_array_almost_equal(result.data, answer.data)

    def test_altitude_tail_after_lift(self):
        talt = AltitudeTail()
        talt.derive(Parameter('Altitude Radio', np.ma.array([0, 5])),
                    Parameter('Pitch', np.ma.array([0, 18])),
                    Attribute('Ground To Lowest Point Of Tail', 10.0/METRES_TO_FEET),
                    Attribute('Main Gear To Lowest Point Of Tail', 35.0/METRES_TO_FEET))
        result = talt.array
        # Lift 5ft
        answer = np.ma.array(data=[10, 5 - 0.815594803123],
                             dtype=np.float, mask=False)
        np.testing.assert_array_almost_equal(result.data, answer.data)

class TestBrakePressure(unittest.TestCase):
    def test_can_operate(self):
        two_sources = ('Brake (L) Press', 'Brake (R) Press')
        four_sources = ('Brake (L) Inboard Press',
                        'Brake (L) Outboard Press',
                        'Brake (R) Inboard Press',
                        'Brake (R) Outboard Press')
        opts = BrakePressure.get_operational_combinations()
        self.assertTrue(two_sources in opts)
        self.assertTrue(four_sources in opts)
        
    def test_basic_two_params(self):
        brake_left = P('Brake (L) Press', np.ma.array([0,1,0,0,0]))
        brake_right = P('Brake (R) Press', np.ma.array([0,0,0,1,0]))
        brakes = BrakePressure()
        brakes.derive(brake_left, brake_right)
        expected = np.ma.array([0.0, 0.5, 0.5, 0.0, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0],
                               mask = [0,0,0,0,0,0,0,0,0,1])        
        np.testing.assert_array_equal(brakes.array, expected)
        
    def test_basic_four_params(self):
        brake_li = P('Brake (L) Inboard Press', np.ma.array([0,0.75,1,0.75,0]))
        brake_lo = P('Brake (L) Outboard Press', np.ma.array([0,0.75,1,0.75,0]))
        brake_ri = P('Brake (R) Inboard Press', np.ma.array([0,0.75,1,0.75,0]))
        brake_ro = P('Brake (R) Outboard Press', np.ma.array([0,0.75,1,0.75,0]))
        brakes = BrakePressure()
        brakes.derive(None, None, brake_li, brake_lo, brake_ri, brake_ro)
        expected = np.ma.array([0.0, 0.5, 0.5, 0.0, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0],
                               mask = [0,0,0,0,0,0,0,0,0,1])        
        self.assertAlmostEqual(brakes.array[4], 0.75)
        self.assertAlmostEqual(brakes.array[8], 1.0)

class TestCabinAltitude(unittest.TestCase):
    def test_can_operate(self):
        expected = [('Cabin Press',)]
        opts = CabinAltitude.get_operational_combinations()
        self.assertEqual(opts,expected)
        
    def test_basic(self):
        cp = P(name='Cabin Press', 
               array=np.ma.array([14.696, 10.108, 4.3727, 2.1490]), 
               units=ut.PSI)
        ca = CabinAltitude()
        ca.derive(cp)
        expected = np.ma.array([0.0, 10000, 30000, 45000])
        ma_test.assert_masked_array_almost_equal (ca.array, expected, decimal=-3)
        

class TestClimbForFlightPhases(unittest.TestCase):
    def test_can_operate(self):
        expected = [('Altitude STD Smoothed','Fast')]
        opts = ClimbForFlightPhases.get_operational_combinations()
        self.assertEqual(opts, expected)
        
    def test_climb_for_flight_phases_basic(self):
        up_and_down_data = np.ma.array([0,0,2,5,3,2,5,6,8,0])
        phase_fast = Fast()
        phase_fast.derive(P('Airspeed', np.ma.array([0]+[100]*8+[0])))
        climb = ClimbForFlightPhases()
        climb.derive(Parameter('Altitude STD Smoothed', up_and_down_data), phase_fast)
        expected = np.ma.array([0,0,2,5,0,0,3,4,6,0])
        ma_test.assert_masked_array_approx_equal(climb.array, expected)




class TestControlColumn(unittest.TestCase):

    def setUp(self):
        ccc = np.ma.array(data=[])
        self.ccc = P('Control Column (Capt)', ccc)
        ccf = np.ma.array(data=[])
        self.ccf = P('Control Column (FO)', ccf)

    def test_can_operate(self):
        expected = [('Control Column (Capt)', 'Control Column (FO)')]
        opts = ControlColumn.get_operational_combinations()
        self.assertEqual(opts, expected)

    @patch('analysis_engine.derived_parameters.blend_two_parameters')
    def test_control_column(self, blend_two_parameters):
        blend_two_parameters.return_value = [None, None, None]
        cc = ControlColumn()
        cc.derive(self.ccc, self.ccf)
        blend_two_parameters.assert_called_once_with(self.ccc, self.ccf)


class TestControlColumnForce(unittest.TestCase):

    def test_can_operate(self):
        expected = [('Control Column Force (Capt)',
                     'Control Column Force (FO)')]
        opts = ControlColumnForce.get_operational_combinations()
        self.assertEqual(opts, expected)

    def test_control_column_force(self):
        ccf = ControlColumnForce()
        ccf.derive(
            ControlColumnForce('Control Column Force (Capt)', np.ma.arange(8)),
            ControlColumnForce('Control Column Force (FO)', np.ma.arange(8)))
        np.testing.assert_array_almost_equal(ccf.array, np.ma.arange(0, 16, 2))


class TestControlWheel(unittest.TestCase):

    def setUp(self):
        cwc = np.ma.array(data=[])
        self.cwc = P('Control Wheel (Capt)', cwc)
        cwf = np.ma.array(data=[])
        self.cwf = P('Control Wheel (FO)', cwf)

    def test_can_operate(self):
        expected = ('Control Wheel (Capt)', 
                    'Control Wheel (FO)', 
                    'Control Wheel Synchro',
                    'Control Wheel Potentiometer')
        opts = ControlWheel.get_operational_combinations()
        self.assertIn(('Control Wheel Synchro',), opts)
        self.assertIn(('Control Wheel Potentiometer',), opts)
        self.assertIn(('Control Wheel (Capt)', 'Control Wheel (FO)'), opts)
        self.assertEqual(opts[-1], expected)
        self.assertEqual(len(opts), 13)

    @patch('analysis_engine.derived_parameters.blend_two_parameters')
    def test_control_wheel(self, blend_two_parameters):
        blend_two_parameters.return_value = [None, None, None]
        cw = ControlWheel()
        cw.derive(self.cwc, self.cwf)
        blend_two_parameters.assert_called_once_with(self.cwc, self.cwf)


class TestControlWheelForce(unittest.TestCase):

    def test_can_operate(self):
        expected = [('Control Wheel Force (Capt)',
                     'Control Wheel Force (FO)')]
        opts = ControlWheelForce.get_operational_combinations()
        self.assertEqual(opts, expected)

    def test_control_wheel_force(self):
        ccf = ControlWheelForce()
        ccf.derive(
            ControlWheelForce('Control Wheel Force (Capt)', np.ma.arange(10)),
            ControlWheelForce('Control Wheel Force (FO)', np.ma.arange(10)))
        np.testing.assert_array_almost_equal(ccf.array, np.ma.arange(0, 20, 2))



class TestDescendForFlightPhases(unittest.TestCase):
    def test_can_operate(self):
        expected = [('Altitude STD Smoothed', 'Fast')]
        opts = DescendForFlightPhases.get_operational_combinations()
        self.assertEqual(opts, expected)
        
    def test_descend_for_flight_phases_basic(self):
        down_and_up_data = np.ma.array([0,0,12,5,3,12,15,10,7,0])
        phase_fast = Fast()
        phase_fast.derive(P('Airspeed', np.ma.array([0]+[100]*8+[0])))
        descend = DescendForFlightPhases()
        descend.derive(Parameter('Altitude STD Smoothed', down_and_up_data ), phase_fast)
        expected = np.ma.array([0,0,0,-7,-9,0,0,-5,-8,0])
        ma_test.assert_masked_array_approx_equal(descend.array, expected)


class TestSidestickAngleCapt(NodeTest, unittest.TestCase):
    def setUp(self):
        self.node_class = SidestickAngleCapt
        self.operational_combinations = [
            ('Sidestick Pitch (Capt)', 'Sidestick Roll (Capt)'),
        ]

    def test_derive(self):
        pitch_array = np.ma.arange(20)
        roll_array = pitch_array[::-1]
        pitch = P('Sidestick Pitch (Capt)', pitch_array)
        roll = P('Sidestick Roll (Capt)', roll_array)
        node = self.node_class()
        node.derive(pitch, roll)

        expected_array = np.ma.sqrt(pitch_array ** 2 + roll_array ** 2)
        np.testing.assert_array_equal(node.array, expected_array)

    def test_derive_from_hdf(self):
        [pitch, roll, sidestick], phase = self.get_params_from_hdf(
            os.path.join(test_data_path, 'dual_input.hdf5'),
            ['Pitch Command (Capt)', 'Roll Command (Capt)', # old names
             self.node_class.get_name()])

        roll.array = align(roll, pitch)

        node = self.node_class()
        node.derive(pitch, roll)
        expected_array = np.ma.sqrt(pitch.array ** 2 + roll.array ** 2)
        np.testing.assert_array_equal(node.array, expected_array)

        np.testing.assert_array_equal(node.array, sidestick.array)


class TestSidestickAngleFO(NodeTest, unittest.TestCase):
    def setUp(self):
        self.node_class = SidestickAngleFO
        self.operational_combinations = [
            ('Sidestick Pitch (FO)', 'Sidestick Roll (FO)'),
        ]

    def test_derive(self):
        pitch_array = np.ma.arange(20)
        roll_array = pitch_array[::-1]
        pitch = P('Sidestick Pitch (FO)', pitch_array)
        roll = P('Sidestick Roll (FO)', roll_array)
        node = self.node_class()
        node.derive(pitch, roll)

        expected_array = np.ma.sqrt(pitch_array ** 2 + roll_array ** 2)
        np.testing.assert_array_equal(node.array, expected_array)

    def test_derive_from_hdf(self):
        [pitch, roll, sidestick], phase = self.get_params_from_hdf(
            os.path.join(test_data_path, 'dual_input.hdf5'),
            ['Pitch Command (FO)', 'Roll Command (FO)',  # old names
             self.node_class.get_name()])

        roll.array = align(roll, pitch)

        node = self.node_class()
        node.derive(pitch, roll)
        expected_array = np.ma.sqrt(pitch.array ** 2 + roll.array ** 2)
        np.testing.assert_array_equal(node.array, expected_array)

        np.testing.assert_array_almost_equal(node.array, sidestick.array)


class TestDistanceToLanding(unittest.TestCase):
    
    def test_can_operate(self):
        expected = [('Distance Travelled', 'Touchdown')]
        opts = DistanceToLanding.get_operational_combinations()
        self.assertEqual(opts, expected)
    
    def test_derive(self):
        distance_travelled = P('Distance Travelled', array=np.ma.arange(0, 100))
        tdwns = KTI('Touchdown', items=[KeyTimeInstance(90, 'Touchdown'),
                                        KeyTimeInstance(95, 'Touchdown')])
        
        expected_result = np.ma.concatenate((np.ma.arange(95, 0, -1),np.ma.arange(0, 5, 1)))
        dtl = DistanceToLanding()
        dtl.derive(distance_travelled, tdwns)
        ma_test.assert_array_equal(dtl.array, expected_result)


class TestDistanceTravelled(unittest.TestCase):
    
    def test_can_operate(self):
        expected = [('Groundspeed',)]
        opts = DistanceTravelled.get_operational_combinations()
        self.assertEqual(opts, expected)

    @patch('analysis_engine.derived_parameters.integrate')
    def test_derive(self, integrate):
        gndspeed = Mock()
        gndspeed.array = Mock()
        gndspeed.frequency = Mock()
        DistanceTravelled().derive(gndspeed)
        integrate.assert_called_once_with(gndspeed.array, gndspeed.frequency,
                                          scale=1.0 / 3600)


class TestDrift(unittest.TestCase):
    
    def test_can_operate(self):
        self.assertTrue(Drift.can_operate(('Drift (1)',)))
        self.assertTrue(Drift.can_operate(('Drift (2)',)))
        self.assertTrue(Drift.can_operate(('Drift (1)', 'Drift (2)')))
        self.assertTrue(Drift.can_operate(('Track', 'Heading')))
    
    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestEng_EPRAvg(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = Eng_EPRAvg
        self.operational_combinations = [
            ('Eng (1) EPR',),
            ('Eng (1) EPR', 'Eng (2) EPR',),
            ('Eng (1) EPR', 'Eng (2) EPR', 'Eng (3) EPR',),
            ('Eng (1) EPR', 'Eng (2) EPR', 'Eng (3) EPR', 'Eng (4) EPR',),
        ]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestEng_EPRMax(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = Eng_EPRMax
        self.operational_combinations = [
            ('Eng (1) EPR',),
            ('Eng (1) EPR', 'Eng (2) EPR',),
            ('Eng (1) EPR', 'Eng (2) EPR', 'Eng (3) EPR',),
            ('Eng (1) EPR', 'Eng (2) EPR', 'Eng (3) EPR', 'Eng (4) EPR',),
        ]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestEng_EPRMin(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = Eng_EPRMin
        self.operational_combinations = [
            ('Eng (1) EPR',),
            ('Eng (1) EPR', 'Eng (2) EPR',),
            ('Eng (1) EPR', 'Eng (2) EPR', 'Eng (3) EPR',),
            ('Eng (1) EPR', 'Eng (2) EPR', 'Eng (3) EPR', 'Eng (4) EPR',),
        ]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestEng_EPRMinFor5Sec(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = Eng_EPRMinFor5Sec
        self.operational_combinations = [('Eng (*) EPR Min',)]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestEng_N1Avg(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = Eng_N1Avg
        self.operational_combinations = [
            ('Eng (1) N1',),
            ('Eng (1) N1', 'Eng (2) N1',),
            ('Eng (1) N1', 'Eng (2) N1', 'Eng (3) N1',),
            ('Eng (1) N1', 'Eng (2) N1', 'Eng (3) N1', 'Eng (4) N1',),
        ]

    def test_derive_two_engines(self):
        # this tests that average is performed on incomplete dependencies and 
        # more than one dependency provided.
        a = np.ma.array(range(0, 10))
        b = np.ma.array(range(10,20))
        a[0] = np.ma.masked
        b[0] = np.ma.masked
        b[-1] = np.ma.masked
        eng_avg = Eng_N1Avg()
        eng_avg.derive(P('a',a), P('b',b), None, None)
        ma_test.assert_array_equal(
            np.ma.filled(eng_avg.array, fill_value=999),
            np.array([999, # both masked, so filled with 999
                      6,7,8,9,10,11,12,13, # unmasked avg of two engines
                      9]) # only second engine value masked
        )


class TestEng_N1Max(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = Eng_N1Max
        self.operational_combinations = [
            ('Eng (1) N1',),
            ('Eng (1) N1', 'Eng (2) N1',),
            ('Eng (1) N1', 'Eng (2) N1', 'Eng (3) N1',),
            ('Eng (1) N1', 'Eng (2) N1', 'Eng (3) N1', 'Eng (4) N1',),
        ]

    def test_derive_two_engines(self):
        # this tests that average is performed on incomplete dependencies and 
        # more than one dependency provided.
        a = np.ma.array(range(0, 10))
        b = np.ma.array(range(10,20))
        a[0] = np.ma.masked
        b[0] = np.ma.masked
        b[-1] = np.ma.masked
        eng = Eng_N1Max()
        eng.derive(P('a',a), P('b',b), None, None)
        ma_test.assert_array_equal(
            np.ma.filled(eng.array, fill_value=999),
            np.array([999, # both masked, so filled with 999
                      11,12,13,14,15,16,17,18,9])
        )
        
    def test_derive_two_engines_offset(self):
        # this tests that average is performed on data sampled alternately.
        a = np.ma.array(range(50, 55))
        b = np.ma.array(range(54, 49, -1)) + 0.2
        eng = Eng_N1Max()
        eng.derive(P('Eng (1)',a,offset=0.25), P('Eng (2)',b, offset=0.75), None, None)
        ma_test.assert_array_equal(eng.array,np.ma.array([54.2, 53.2, 52.2, 53, 54]))
        self.assertEqual(eng.offset, 0)
        
        
class TestEng_N1Min(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = Eng_N1Min
        self.operational_combinations = [
            ('Eng (1) N1',),
            ('Eng (1) N1', 'Eng (2) N1',),
            ('Eng (1) N1', 'Eng (2) N1', 'Eng (3) N1',),
            ('Eng (1) N1', 'Eng (2) N1', 'Eng (3) N1', 'Eng (4) N1',),
        ]

    def test_derive_two_engines(self):
        # this tests that average is performed on incomplete dependencies and 
        # more than one dependency provided.
        a = np.ma.array(range(0, 10))
        b = np.ma.array(range(10,20))
        a[0] = np.ma.masked
        b[0] = np.ma.masked
        b[-1] = np.ma.masked
        eng = Eng_N1Min()
        eng.derive(P('a',a), P('b',b), None, None)
        ma_test.assert_array_equal(
            np.ma.filled(eng.array, fill_value=999),
            np.array([999, # both masked, so filled with 999
                      1,2,3,4,5,6,7,8,9])
        )


class TestEng_N1MinFor5Sec(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = Eng_N1MinFor5Sec
        self.operational_combinations = [('Eng (*) N1 Min',)]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestEng_N2Avg(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = Eng_N2Avg
        self.operational_combinations = [
            ('Eng (1) N2',),
            ('Eng (1) N2', 'Eng (2) N2',),
            ('Eng (1) N2', 'Eng (2) N2', 'Eng (3) N2',),
            ('Eng (1) N2', 'Eng (2) N2', 'Eng (3) N2', 'Eng (4) N2',),
        ]

    def test_derive_two_engines(self):
        # this tests that average is performed on incomplete dependencies and 
        # more than one dependency provided.
        a = np.ma.array(range(0, 10))
        b = np.ma.array(range(10,20))
        a[0] = np.ma.masked
        b[0] = np.ma.masked
        b[-1] = np.ma.masked
        eng_avg = Eng_N2Avg()
        eng_avg.derive(P('a',a), P('b',b), None, None)
        ma_test.assert_array_equal(
            np.ma.filled(eng_avg.array, fill_value=999),
            np.array([999, # both masked, so filled with 999
                      6,7,8,9,10,11,12,13, # unmasked avg of two engines
                      9]) # only second engine value masked
        )


class TestEng_N2Max(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = Eng_N2Max
        self.operational_combinations = [
            ('Eng (1) N2',),
            ('Eng (1) N2', 'Eng (2) N2',),
            ('Eng (1) N2', 'Eng (2) N2', 'Eng (3) N2',),
            ('Eng (1) N2', 'Eng (2) N2', 'Eng (3) N2', 'Eng (4) N2',),
        ]

    def test_derive_two_engines(self):
        # this tests that average is performed on incomplete dependencies and 
        # more than one dependency provided.
        a = np.ma.array(range(0, 10))
        b = np.ma.array(range(10,20))
        a[0] = np.ma.masked
        b[0] = np.ma.masked
        b[-1] = np.ma.masked
        eng = Eng_N2Max()
        eng.derive(P('a',a), P('b',b), None, None)
        ma_test.assert_array_equal(
            np.ma.filled(eng.array, fill_value=999),
            np.array([999, # both masked, so filled with 999
                      11,12,13,14,15,16,17,18,9])
        )


class TestEng_N2Min(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = Eng_N2Min
        self.operational_combinations = [
            ('Eng (1) N2',),
            ('Eng (1) N2', 'Eng (2) N2',),
            ('Eng (1) N2', 'Eng (2) N2', 'Eng (3) N2',),
            ('Eng (1) N2', 'Eng (2) N2', 'Eng (3) N2', 'Eng (4) N2',),
        ]

    def test_derive_two_engines(self):
        # this tests that average is performed on incomplete dependencies and 
        # more than one dependency provided.
        a = np.ma.array(range(0, 10))
        b = np.ma.array(range(10,20))
        a[0] = np.ma.masked
        b[0] = np.ma.masked
        b[-1] = np.ma.masked
        eng = Eng_N2Min()
        eng.derive(P('a',a), P('b',b), None, None)
        ma_test.assert_array_equal(
            np.ma.filled(eng.array, fill_value=999),
            np.array([999, # both masked, so filled with 999
                      1,2,3,4,5,6,7,8,9])
        )


class TestEng_N3Avg(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = Eng_N3Avg
        self.operational_combinations = [
            ('Eng (1) N3',),
            ('Eng (1) N3', 'Eng (2) N3',),
            ('Eng (1) N3', 'Eng (2) N3', 'Eng (3) N3',),
            ('Eng (1) N3', 'Eng (2) N3', 'Eng (3) N3', 'Eng (4) N3',),
        ]

    def test_derive_two_engines(self):
        # this tests that average is performed on incomplete dependencies and
        # more than one dependency provided.
        a = np.ma.array(range(0, 10))
        b = np.ma.array(range(10,20))
        a[0] = np.ma.masked
        b[0] = np.ma.masked
        b[-1] = np.ma.masked
        eng_avg = Eng_N3Avg()
        eng_avg.derive(P('a',a), P('b',b), None, None)
        ma_test.assert_array_equal(
            np.ma.filled(eng_avg.array, fill_value=999),
            np.array([999, # both masked, so filled with 999
                      6,7,8,9,10,11,12,13, # unmasked avg of two engines
                      9]) # only second engine value masked
        )


class TestEng_N3Max(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = Eng_N3Max
        self.operational_combinations = [
            ('Eng (1) N3',),
            ('Eng (1) N3', 'Eng (2) N3',),
            ('Eng (1) N3', 'Eng (2) N3', 'Eng (3) N3',),
            ('Eng (1) N3', 'Eng (2) N3', 'Eng (3) N3', 'Eng (4) N3',),
        ]

    def test_derive_two_engines(self):
        # this tests that average is performed on incomplete dependencies and
        # more than one dependency provided.
        a = np.ma.array(range(0, 10))
        b = np.ma.array(range(10,20))
        a[0] = np.ma.masked
        b[0] = np.ma.masked
        b[-1] = np.ma.masked
        eng = Eng_N3Max()
        eng.derive(P('a',a), P('b',b), None, None)
        ma_test.assert_array_equal(
            np.ma.filled(eng.array, fill_value=999),
            np.array([999, # both masked, so filled with 999
                      11,12,13,14,15,16,17,18,9])
        )


class TestEng_N3Min(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = Eng_N3Min
        self.operational_combinations = [
            ('Eng (1) N3',),
            ('Eng (1) N3', 'Eng (2) N3',),
            ('Eng (1) N3', 'Eng (2) N3', 'Eng (3) N3',),
            ('Eng (1) N3', 'Eng (2) N3', 'Eng (3) N3', 'Eng (4) N3',),
        ]

    def test_derive_two_engines(self):
        # this tests that average is performed on incomplete dependencies and
        # more than one dependency provided.
        a = np.ma.array(range(0, 10))
        b = np.ma.array(range(10,20))
        a[0] = np.ma.masked
        b[0] = np.ma.masked
        b[-1] = np.ma.masked
        eng = Eng_N3Min()
        eng.derive(P('a',a), P('b',b), None, None)
        ma_test.assert_array_equal(
            np.ma.filled(eng.array, fill_value=999),
            np.array([999, # both masked, so filled with 999
                      1,2,3,4,5,6,7,8,9])
        )


class TestEng_NpAvg(unittest.TestCase):
    def test_can_operate(self):
        opts = Eng_NpAvg.get_operational_combinations()
        self.assertEqual(opts[0], ('Eng (1) Np',))
        self.assertEqual(opts[-1], ('Eng (1) Np', 'Eng (2) Np', 'Eng (3) Np', 'Eng (4) Np'))
        self.assertEqual(len(opts), 15) # 15 combinations accepted!


    def test_derive_two_engines(self):
        # this tests that average is performed on incomplete dependencies and
        # more than one dependency provided.
        a = np.ma.array(range(0, 10))
        b = np.ma.array(range(10,20))
        a[0] = np.ma.masked
        b[0] = np.ma.masked
        b[-1] = np.ma.masked
        eng_avg = Eng_NpAvg()
        eng_avg.derive(P('a',a), P('b',b), None, None)
        ma_test.assert_array_equal(
            np.ma.filled(eng_avg.array, fill_value=999),
            np.array([999, # both masked, so filled with 999
                      6,7,8,9,10,11,12,13, # unmasked avg of two engines
                      9]) # only second engine value masked
        )


class TestEng_NpMax(unittest.TestCase):
    def test_can_operate(self):
        opts = Eng_NpMax.get_operational_combinations()
        self.assertEqual(opts[0], ('Eng (1) Np',))
        self.assertEqual(opts[-1], ('Eng (1) Np', 'Eng (2) Np', 'Eng (3) Np', 'Eng (4) Np'))
        self.assertEqual(len(opts), 15) # 15 combinations accepted!

    def test_derive_two_engines(self):
        # this tests that average is performed on incomplete dependencies and
        # more than one dependency provided.
        a = np.ma.array(range(0, 10))
        b = np.ma.array(range(10,20))
        a[0] = np.ma.masked
        b[0] = np.ma.masked
        b[-1] = np.ma.masked
        eng = Eng_NpMax()
        eng.derive(P('a',a), P('b',b), None, None)
        ma_test.assert_array_equal(
            np.ma.filled(eng.array, fill_value=999),
            np.array([999, # both masked, so filled with 999
                      11,12,13,14,15,16,17,18,9])
        )


class TestEng_NpMin(unittest.TestCase):
    def test_can_operate(self):
        opts = Eng_NpMin.get_operational_combinations()
        self.assertEqual(opts[0], ('Eng (1) Np',))
        self.assertEqual(opts[-1], ('Eng (1) Np', 'Eng (2) Np', 'Eng (3) Np', 'Eng (4) Np'))
        self.assertEqual(len(opts), 15) # 15 combinations accepted!

    def test_derive_two_engines(self):
        # this tests that average is performed on incomplete dependencies and
        # more than one dependency provided.
        a = np.ma.array(range(0, 10))
        b = np.ma.array(range(10,20))
        a[0] = np.ma.masked
        b[0] = np.ma.masked
        b[-1] = np.ma.masked
        eng = Eng_NpMin()
        eng.derive(P('a',a), P('b',b), None, None)
        ma_test.assert_array_equal(
            np.ma.filled(eng.array, fill_value=999),
            np.array([999, # both masked, so filled with 999
                      1,2,3,4,5,6,7,8,9])
        )


class TestFuelQty(unittest.TestCase):
    def test_can_operate(self):
        # testing for number of combinations possible, will operate with at
        # least one of the listed parameters. Not listing all operational
        # combinations as this can get very large (2**n-1) where n is the
        # number of parameters (-1 as none is not a option)
        self.assertEqual(len(FuelQty.get_operational_combinations()), 2**8-1)
    
    def test_three_tanks(self):
        fuel_qty1 = P('Fuel Qty (L)', 
                      array=np.ma.array([1,2,3], mask=[False, False, False]))
        fuel_qty2 = P('Fuel Qty (C)', 
                      array=np.ma.array([2,4,6], mask=[False, False, False]))
        # Mask will be interpolated by repair_mask.
        fuel_qty3 = P('Fuel Qty (R)',
                      array=np.ma.array([3,6,9], mask=[False, True, False]))
        fuel_qty_node = FuelQty()
        fuel_qty_node.derive(fuel_qty1, fuel_qty2, None, None, fuel_qty3,
                             None, None, None)
        np.testing.assert_array_equal(fuel_qty_node.array,
                                      np.ma.array([6, 12, 18]))
        # Works without all parameters.
        fuel_qty_node.derive(fuel_qty1, *[None,]*7)
        np.testing.assert_array_equal(fuel_qty_node.array,
                                      np.ma.array([1, 2, 3]))

    def test_four_tanks(self):
        fuel_qty1 = P('Fuel Qty (L)', 
                      array=np.ma.array([1,2,3], mask=[False, False, False]))
        fuel_qty2 = P('Fuel Qty (C)', 
                      array=np.ma.array([2,4,6], mask=[False, False, False]))
        # Mask will be interpolated by repair_mask.
        fuel_qty3 = P('Fuel Qty (R)',
                      array=np.ma.array([3,6,9], mask=[False, True, False]))
        fuel_qty_a = P('Fuel Qty (Aux)',
                      array=np.ma.array([11,12,13], mask=[False, False, False]))
        fuel_qty_node = FuelQty()
        fuel_qty_node.derive(fuel_qty1, fuel_qty2, None, None, fuel_qty3,
                             fuel_qty_a, None, None)
        np.testing.assert_array_equal(fuel_qty_node.array,
                                      np.ma.array([17, 24, 31]))
    
    def test_masked_tank(self):
        fuel_qty1 = P('Fuel Qty (L)', 
                      array=np.ma.array([1,2,3], mask=[False, False, False]))
        fuel_qty2 = P('Fuel Qty (R)', 
                      array=np.ma.array([2,4,6], mask=[True, True, True]))
        # Mask will be interpolated by repair_mask.
        fuel_qty_node = FuelQty()
        fuel_qty_node.derive(fuel_qty1, None, None, None, fuel_qty2, None, None, None)
        np.testing.assert_array_equal(fuel_qty_node.array,
                                      np.ma.array([1, 2, 3]))


class TestFuelQtyL(unittest.TestCase):
    def test_can_operate(self):
        # testing for number of combinations possible, will operate with at
        # least one of the listed parameters. Not listing all operational
        # combinations as this can get very large (2**n-1) where n is the
        # number of parameters (-1 as none is not a option)
        opts = FuelQtyL.get_operational_combinations()
        self.assertTrue(('Fuel Qty (L1)',) in opts)
        self.assertTrue(('Fuel Qty (L2)',) in opts)
        self.assertTrue(('Fuel Qty (L3)',) in opts)
        self.assertTrue(('Fuel Qty (L1)', 'Fuel Qty (L2)', 'Fuel Qty (L3)') in opts)
    
    def test_three_tanks(self):
        fuel_qty1 = P('Fuel Qty (L1)', 
                      array=np.ma.array([1,2,3], mask=[False, False, False]))
        fuel_qty2 = P('Fuel Qty (L2)', 
                      array=np.ma.array([2,4,6], mask=[False, False, False]))
        fuel_qty3 = P('Fuel Qty (L3)',
                      array=np.ma.array([3,6,9], mask=[False, True, False]))
        fuel_qty_node = FuelQtyL()
        fuel_qty_node.derive(fuel_qty1, None, None)
        np.testing.assert_array_equal(fuel_qty_node.array,
                                      np.ma.array([1, 2, 3]))
        fuel_qty_node.derive(None, fuel_qty2, None)
        np.testing.assert_array_equal(fuel_qty_node.array,
                                      np.ma.array([2, 4, 6]))
        fuel_qty_node.derive(None, fuel_qty2, fuel_qty3)
        np.testing.assert_array_equal(fuel_qty_node.array,
                                      np.ma.array([5, 10, 15],
                                                  mask=[False, True, False]))
        fuel_qty_node.derive(fuel_qty1, fuel_qty2, fuel_qty3)
        np.testing.assert_array_equal(fuel_qty_node.array,
                                      np.ma.array([6, 12, 18],
                                                  mask=[False, True, False]))


class TestFuelQtyR(unittest.TestCase):
    def test_can_operate(self):
        # testing for number of combinations possible, will operate with at
        # least one of the listed parameters. Not listing all operational
        # combinations as this can get very large (2**n-1) where n is the
        # number of parameters (-1 as none is not a option)
        opts = FuelQtyR.get_operational_combinations()
        self.assertTrue(('Fuel Qty (R1)',) in opts)
        self.assertTrue(('Fuel Qty (R2)',) in opts)
        self.assertTrue(('Fuel Qty (R3)',) in opts)
        self.assertTrue(('Fuel Qty (R1)', 'Fuel Qty (R2)', 'Fuel Qty (R3)') in opts)
    
    def test_three_tanks(self):
        fuel_qty1 = P('Fuel Qty (R1)', 
                      array=np.ma.array([3,2,1], mask=[False, False, False]))
        fuel_qty2 = P('Fuel Qty (R2)', 
                      array=np.ma.array([6,4,2], mask=[False, False, False]))
        fuel_qty3 = P('Fuel Qty (R3)',
                      array=np.ma.array([9,6,3], mask=[False, True, False]))
        fuel_qty_node = FuelQtyR()
        fuel_qty_node.derive(fuel_qty1, None, None)
        np.testing.assert_array_equal(fuel_qty_node.array,
                                      np.ma.array([3, 2, 1]))
        fuel_qty_node.derive(None, fuel_qty2, None)
        np.testing.assert_array_equal(fuel_qty_node.array,
                                      np.ma.array([6, 4, 2]))
        fuel_qty_node.derive(None, fuel_qty2, fuel_qty3)
        np.testing.assert_array_equal(fuel_qty_node.array,
                                      np.ma.array([15, 10, 5],
                                                  mask=[False, True, False]))
        fuel_qty_node.derive(fuel_qty1, fuel_qty2, fuel_qty3)
        np.testing.assert_array_equal(fuel_qty_node.array,
                                      np.ma.array([18, 12, 6],
                                                  mask=[False, True, False]))


class TestGrossWeightSmoothed(unittest.TestCase):
    def test_gw_real_data_1(self):
        ff = load(os.path.join(test_data_path,
                               'gross_weight_smoothed_1_ff.nod'))
        gw = load(os.path.join(test_data_path,
                               'gross_weight_smoothed_1_gw.nod'))
        gw_orig = gw.array.copy()
        climbs = load(os.path.join(test_data_path,
                                   'gross_weight_smoothed_1_climbs.nod'))
        descends = load(os.path.join(test_data_path,
                                     'gross_weight_smoothed_1_descends.nod'))
        fast = load(os.path.join(test_data_path,
                                 'gross_weight_smoothed_1_fast.nod'))
        gws = GrossWeightSmoothed()
        gws.derive(ff, gw, climbs, descends, fast)
        # Start is similar.
        self.assertTrue(abs(gws.array[640] - gw_orig[640]) < 30)
        # Climbing diverges.
        self.assertTrue(abs(gws.array[1150] - gw_orig[1150]) < 260)
        # End is similar.
        self.assertTrue(abs(gws.array[2500] - gw_orig[2500]) < 30)
        
    def test_gw_real_data_2(self): 
        ff = load(os.path.join(test_data_path,
                               'gross_weight_smoothed_2_ff.nod'))
        gw = load(os.path.join(test_data_path,
                               'gross_weight_smoothed_2_gw.nod'))
        gw_orig = gw.array.copy()
        climbs = load(os.path.join(test_data_path,
                                   'gross_weight_smoothed_2_climbs.nod'))
        descends = load(os.path.join(test_data_path,
                                     'gross_weight_smoothed_2_descends.nod'))
        fast = load(os.path.join(test_data_path,
                                 'gross_weight_smoothed_2_fast.nod'))
        gws = GrossWeightSmoothed()
        gws.derive(ff, gw, climbs, descends, fast)
        # Start is similar.
        self.assertTrue(abs(gws.array[600] - gw_orig[600]) < 35)
        # Climbing diverges.
        self.assertTrue(abs(gws.array[1500] - gw_orig[1500]) < 180)
        # Descending diverges.
        self.assertTrue(abs(gws.array[5800] - gw_orig[5800]) < 120)
    
    def test_gw_masked(self): 
        weight = P('Gross Weight',np.ma.array([292,228,164,100],dtype=float),offset=0.0,frequency=1/64.0)
        fuel_flow = P('Eng (*) Fuel Flow',np.ma.array([3600]*256,dtype=float),offset=0.0,frequency=1.0)
        weight_aligned = align(weight, fuel_flow)
        climb = buildsection('Climbing', 10, 20)
        descend = buildsection('Descending', 40, 50)
        fast = buildsection('Fast', None, None)
        gws = GrossWeightSmoothed()
        result = gws.get_derived([fuel_flow, weight, climb, descend, fast])  
        ma_test.assert_equal(result.array, weight_aligned)
    
    def test_gw_formula(self):
        weight = P('Gross Weight',np.ma.array([292,228,164,100],dtype=float),offset=0.0,frequency=1/64.0)
        fuel_flow = P('Eng (*) Fuel Flow',np.ma.array([3600]*256,dtype=float),offset=0.0,frequency=1.0)
        climb = buildsection('Climbing', 10, 20)
        descend = buildsection('Descending', 40, 50)
        fast = buildsection('Fast', 10, len(fuel_flow.array))
        gws = GrossWeightSmoothed()
        result = gws.get_derived([fuel_flow, weight, climb, descend, fast])
        self.assertEqual(result.array[0], 292.0)
        self.assertEqual(result.array[-1], 37.0)
        
    def test_gw_formula_with_many_samples(self):
        weight = P('Gross Weight',np.ma.array(data=range(56400, 50000, -64), 
                                              mask=False, dtype=float),
                   offset=0.0, frequency=1 / 64.0)
        fuel_flow = P('Eng (*) Fuel Flow', np.ma.array([3600] * 64 * 100,
                                                       dtype=float),
                      offset=0.0, frequency=1.0)
        climb = buildsection('Climbing', 10, 20)
        descend = buildsection('Descending', 50, 60)
        fast = buildsection('Fast', 10, len(fuel_flow.array))
        gws = GrossWeightSmoothed()
        result = gws.get_derived([fuel_flow, weight, climb, descend, fast])
        self.assertEqual(result.array[1], 56400-1)
        
    def test_gw_formula_with_good_data(self):
        weight = P('Gross Weight', np.ma.array(data=[484, 420, 356, 292, 228, 164, 100],
                                               mask=[1, 0, 0, 0, 0, 1, 0], dtype=float),
                   offset=0.0, frequency=1 / 64.0)
        fuel_flow = P('Eng (*) Fuel Flow', np.ma.array([3600] * 64 * 7, dtype=float),
                      offset=0.0, frequency=1.0)
        climb = buildsection('Climbing', 10, 20)
        descend = buildsection('Descending', 60, 70)
        fast = buildsection('Fast', 10, len(fuel_flow.array))
        gws = GrossWeightSmoothed()
        result = gws.get_derived([fuel_flow, weight, climb, descend, fast])
        self.assertEqual(result.array[0], 484.0)
        self.assertEqual(result.array[-1], 37.0)
        
    def test_gw_formula_climbing(self):
        weight = P('Gross Weight',np.ma.array(data=[484,420,356,292,228,164,100],
                                              mask=[1,0,0,0,0,1,0],dtype=float),
                   offset=0.0,frequency=1/64.0)
        fuel_flow = P('Eng (*) Fuel Flow',
                      np.ma.array([3600] * 64 * 7, dtype=float))
        climb = buildsection('Climbing', 1, 4)
        descend = buildsection('Descending', 20, 30)
        fast = buildsection('Fast', 10, len(fuel_flow.array))
        gws = GrossWeightSmoothed()
        result = gws.get_derived([fuel_flow, weight, climb, descend, fast])
        self.assertEqual(result.array[0], 484.0)
        self.assertEqual(result.array[-1], 37.0)
        
    def test_gw_descending(self):
        weight = P('Gross Weight',np.ma.array(
            data=[484, 420, 356, 292, 228, 164, 100],
            mask=[1, 0, 0, 0, 0, 1, 0], dtype=float),
                   offset=0.0, frequency=1 / 64.0)
        fuel_flow = P('Eng (*) Fuel Flow',
                      np.ma.array([3600] * 64 * 7, dtype=float),
                      offset=0.0, frequency=1.0)
        gws = GrossWeightSmoothed()
        climb = S('Climbing')
        descend = buildsection('Descending', 3, 5)
        fast = buildsection('Fast', 50, 450)
        gws = GrossWeightSmoothed()
        result = gws.get_derived([fuel_flow, weight, climb, descend, fast])
        self.assertEqual(result.array[0], 484.0)
        self.assertEqual(result.array[-1], 37.0)
        
    def test_gw_one_masked_data_point(self):
        weight = P('Gross Weight',np.ma.array(data=[0],
                                              mask=[1],dtype=float),
                   offset=0.0,frequency=1/64.0)
        fuel_flow = P('Eng (*) Fuel Flow',np.ma.array([0]*64,dtype=float),
                      offset=0.0,frequency=1.0)
        gws = GrossWeightSmoothed()
        climb = S('Climbing')
        descend = S('Descending')
        fast = buildsection('Fast', 0, 1)
        gws = GrossWeightSmoothed()
        gws.get_derived([fuel_flow, weight, climb, descend, fast])
        self.assertEqual(len(gws.array),64)
        self.assertEqual(gws.frequency, fuel_flow.frequency)
        self.assertEqual(gws.offset, fuel_flow.offset)

class TestGroundspeed(unittest.TestCase):
    
    def test_can_operate(self):
        opts = Groundspeed.get_operational_combinations()
        self.assertEqual(opts, [('Groundspeed (1)',), ('Groundspeed (2)',), ('Groundspeed (1)', 'Groundspeed (2)')])
    
    def test_basic(self):
        one = P('Groundspeed (1)', np.ma.array([100,200,300]), frequency=0.5, offset=0.0)
        two = P('Groundspeed (2)', np.ma.array([150,250,350]), frequency=0.5, offset=1.0)
        frame = A('Frame', 'Not DHL')
        gs = Groundspeed()
        gs.derive(one, two)
        # Note: end samples are not 100 & 350 due to method of merging. 
        np.testing.assert_array_equal(gs.array[1:-1], np.array([150, 200, 250, 300]))
        self.assertEqual(gs.frequency, 1.0)
        self.assertEqual(gs.offset, 0.0)
        

class TestGroundspeedAlongTrack(unittest.TestCase):

    @unittest.skip('Commented out until new computation of sliding motion')
    def test_can_operate(self):
        expected = [('Groundspeed','Acceleration Along Track', 'Altitude AAL',
                     'ILS Glideslope')]
        opts = GroundspeedAlongTrack.get_operational_combinations()
        self.assertEqual(opts, expected)

    @unittest.skip('Commented out until new computation of sliding motion')
    def test_groundspeed_along_track_basic(self):
        gat = GroundspeedAlongTrack()
        gspd = P('Groundspeed',np.ma.array(data=[100]*2+[120]*18), frequency=1)
        accel = P('Acceleration Along Track',np.ma.zeros(20), frequency=1)
        gat.derive(gspd, accel)
        # A first order lag of 6 sec time constant rising from 100 to 120
        # will pass through 110 knots between 13 & 14 seconds after the step
        # rise.
        self.assertLess(gat.array[5],56.5)
        self.assertGreater(gat.array[6],56.5)
        
    @unittest.skip('Commented out until new computation of sliding motion')
    def test_groundspeed_along_track_accel_term(self):
        gat = GroundspeedAlongTrack()
        gspd = P('Groundspeed',np.ma.array(data=[100]*200), frequency=1)
        accel = P('Acceleration Along Track',np.ma.ones(200)*.1, frequency=1)
        accel.array[0]=0.0
        gat.derive(gspd, accel)
        # The resulting waveform takes time to start going...
        self.assertLess(gat.array[4],55.0)
        # ...then rises under the influence of the lag...
        self.assertGreater(gat.array[16],56.0)
        # ...to a peak...
        self.assertGreater(np.ma.max(gat.array.data),16)
        # ...and finally decays as the longer washout time constant takes effect.
        self.assertLess(gat.array[199],52.0)


#class TestHeadingContinuous(unittest.TestCase):
    #def test_can_operate(self):
        #expected = [('Heading',)]
        #opts = HeadingContinuous.get_operational_combinations()
        #self.assertEqual(opts, expected)

    #def test_heading_continuous(self):
        #head = HeadingContinuous()
        #head.derive(P('Heading',np.ma.remainder(
            #np.ma.array(range(10))+355,360.0)))
        
        #answer = np.ma.array(data=[355.0, 356.0, 357.0, 358.0, 359.0, 360.0, 
                                   #361.0, 362.0, 363.0, 364.0],
                             #dtype=np.float, mask=False)

        ##ma_test.assert_masked_array_approx_equal(res, answer)
        #np.testing.assert_array_equal(head.array.data, answer.data)

class TestHeadingContinuous(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = HeadingContinuous
        self.operational_combinations = [('Heading',),
                                         ('Heading (Capt)', 'Heading (FO)'),
                                         ('Heading', 'Heading (Capt)',),
                                         ('Heading', 'Heading (FO)'),
                                         ('Heading', 'Heading (Capt)', 'Heading (FO)'),
                                         ('Heading','Frame'),
                                         ('Heading (Capt)', 'Heading (FO)','Frame'),
                                         ('Heading', 'Heading (Capt)','Frame'),
                                         ('Heading', 'Heading (FO)','Frame'),
                                         ('Heading', 'Heading (Capt)', 'Heading (FO)','Frame')
                                         ]

    def test_heading_continuous_basic(self):
        hdg = P('Heading',np.ma.remainder(np.ma.array(range(10))+355,360.0))
        hdg.array[2] = np.ma.masked
        node = self.node_class()
        node.derive(hdg, None, None)
        expected = np.ma.array(data=[355.0, 356.0, 357.0, 358.0, 359.0, 360.0, 
                                     361.0, 362.0, 363.0, 364.0],
                               dtype=np.float, mask=False)
        ma_test.assert_equal(node.array, expected)

    def test_heading_continuous_merged(self):
        hdg = P('Heading',np.ma.remainder(np.ma.array(range(10))+355,360.0))
        hdg_ca = P('Heading (Capt)',np.ma.array([5,6,7,8,9.0]),offset=0.1,frequency=0.5)
        hdg_fo = P('Heading (FO)',np.ma.array([15,16,17,18,19.0]),offset=1.1,frequency=0.5)
        node = self.node_class()
        node.derive(hdg, hdg_ca, hdg_fo)
        expected = np.ma.array(data=np.array(range(10))/2.0+9.75,
                               dtype=np.float, mask=False)
        expected[0]=10.0
        expected[-1]=14.0
        ma_test.assert_equal(node.array, expected)
        self.assertEqual(node.offset, 0.1)
        self.assertEqual(node.frequency, 1,0)

    def test_heading_continuous_merged_rollover(self):
        hdg = P('Heading',np.ma.remainder(np.ma.array(range(10))+355,360.0))
        hdg_ca = P('Heading (Capt)',np.ma.array([358,2,6,10, 14.0]),offset=0.1,frequency=0.5)
        hdg_ca.array[2]=np.ma.masked
        hdg_fo = P('Heading (FO)',np.ma.array([346,350,354,358,2.0]),offset=1.1,frequency=0.5)
        hdg_fo.array[3]=np.ma.masked
        node = self.node_class()
        node.derive(hdg, hdg_ca, hdg_fo)
        expected = np.ma.array(data=np.array(range(10))*2.0+351.0,
                               dtype=np.float, mask=False)
        expected[0]=352.0
        expected[-1]=368.0
        ma_test.assert_equal(node.array, expected)
        self.assertEqual(node.offset, 0.1)
        self.assertEqual(node.frequency, 1,0)

    def test_heading_continuous_not_hercules(self):
        hdg = P('Heading',np.ma.array(data=[10]*60, mask=[0]*20+[1]*20+[0]*20))
        con_hdg = HeadingContinuous()
        con_hdg.derive(hdg, None, None, None)
        # REPAIR_DURATION is limited to 10 seconds, so this should not be repaired.
        self.assertEqual(np.ma.count(con_hdg.array), 40)
        
    def test_heading_continuous_hercules(self):
        hdg = P('Heading',np.ma.array(data=[10]*60, mask=[0]*20+[1]*20+[0]*20))
        con_hdg = HeadingContinuous()
        herc = A('Frame', 'L382-Hercules')
        con_hdg.derive(hdg, None, None, herc)
        # The smoothing algorithm will leave two samples masked at the beginning and end of the array.
        self.assertEqual(np.ma.count(con_hdg.array), 56)
        
    def test_heading_continuous_starting_north(self):
        hdg_ca = P('Heading (Capt)',np.ma.array([358,358,0.0,1.0]))
        hdg_fo = P('Heading (FO)',np.ma.array([0.0,1.0,359,357]),offset=0.5)
        node = self.node_class()
        node.derive(None, hdg_ca, hdg_fo, None)
        expected = np.ma.array([359,359,359.25,360,360,359.75,359.5,359 ])
        ma_test.assert_equal(node.array, expected)


class TestTrack(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = Track
        self.operational_combinations = [('Track Continuous',)]

    def test_derive_basic(self):
        track = Parameter('Track Continuous', array=np.ma.arange(0, 1000, 100))
        node = self.node_class()
        node.derive(track)
        expected = [0, 100, 200, 300, 40, 140, 240, 340, 80, 180]
        ma_test.assert_equal(node.array, expected)


class TestTrackContinuous(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = TrackContinuous
        self.operational_combinations = [('Heading Continuous', 'Drift')]

    def test_derive_basic(self):
        heading = Parameter('Heading Continuous', array=np.ma.arange(0, 100, 10))
        drift = Parameter('Drift', array=np.ma.arange(0, 1, 0.1))
        node = self.node_class()
        node.derive(heading, drift)
        expected = [0, 10.1, 20.2, 30.3, 40.4, 50.5, 60.6, 70.7, 80.8, 90.9]
        ma_test.assert_equal(node.array, expected)


class TestTrackTrue(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = TrackTrue
        self.operational_combinations = [('Track True Continuous',)]

    def test_derive_basic(self):
        track = Parameter('Track True Continuous', array=np.ma.arange(0, 1000, 100))
        node = self.node_class()
        node.derive(track)
        expected = [0, 100, 200, 300, 40, 140, 240, 340, 80, 180]
        ma_test.assert_equal(node.array, expected)


class TestTrackTrueContinuous(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = TrackTrueContinuous
        self.operational_combinations = [('Heading True Continuous', 'Drift')]

    def test_derive_basic(self):
        heading = Parameter('Heading True Continuous', array=np.ma.arange(0, 100, 10))
        drift = Parameter('Drift', array=np.ma.arange(0, 1, 0.1))
        node = self.node_class()
        node.derive(heading, drift)
        expected = [0, 10.1, 20.2, 30.3, 40.4, 50.5, 60.6, 70.7, 80.8, 90.9]
        ma_test.assert_equal(node.array, expected)

    def test_derive_extra(self):
        # Compare IRU Track Angle True (recorded) against the derived:
        heading = load(os.path.join(test_data_path, 'HeadingTrack_Heading_True.nod'))
        drift = load(os.path.join(test_data_path, 'HeadingTrack_Drift.nod'))
        node = self.node_class()
        node.derive(heading, drift)
        expected = load(os.path.join(test_data_path, 'HeadingTrack_IRU_Track_Angle_Recorded.nod'))
        assert_array_within_tolerance(node.array % 360, expected.array, 10, 98)


class TestTrackDeviationFromRunway(unittest.TestCase):
    def test_can_operate(self):
        self.assertEqual(
            TrackDeviationFromRunway.get_operational_combinations(),
            [('Track True Continuous', 'FDR Takeoff Runway'),
             ('Track True Continuous', 'Approach Information'),
             ('Track Continuous', 'FDR Takeoff Runway'),
             ('Track Continuous', 'Approach Information'),
             ('Track True Continuous', 'Track Continuous', 'FDR Takeoff Runway'),
             ('Track True Continuous', 'Track Continuous', 'Approach Information'),
             ('Track True Continuous', 'Takeoff', 'FDR Takeoff Runway'),
             ('Track True Continuous', 'Takeoff', 'Approach Information'),
             ('Track True Continuous', 'FDR Takeoff Runway', 'Approach Information'),
             ('Track Continuous', 'Takeoff', 'FDR Takeoff Runway'),
             ('Track Continuous', 'Takeoff', 'Approach Information'),
             ('Track Continuous', 'FDR Takeoff Runway', 'Approach Information'),
             ('Track True Continuous', 'Track Continuous', 'Takeoff', 'FDR Takeoff Runway'),
             ('Track True Continuous', 'Track Continuous', 'Takeoff', 'Approach Information'),
             ('Track True Continuous', 'Track Continuous', 'FDR Takeoff Runway', 'Approach Information'),
             ('Track True Continuous', 'Takeoff', 'FDR Takeoff Runway', 'Approach Information'),
             ('Track Continuous', 'Takeoff', 'FDR Takeoff Runway', 'Approach Information'),
             ('Track True Continuous', 'Track Continuous', 'Takeoff', 'FDR Takeoff Runway', 'Approach Information')]
        )
        
    def test_deviation(self):
        apps = App(items=[ApproachItem(
            'LANDING', slice(8763, 9037),
            airport={'code': {'iata': 'FRA', 'icao': 'EDDF'},
                     'distance': 2.2981699358981746,
                     'id': 2289,
                     'latitude': 50.0264,
                     'location': {'city': 'Frankfurt-Am-Main',
                                  'country': 'Germany'},
                     'longitude': 8.54313,
                     'magnetic_variation': 'E000459 0106',
                     'name': 'Frankfurt Am Main'},
            runway={'end': {'latitude': 50.027542, 'longitude': 8.534175},
                    'glideslope': {'angle': 3.0,
                                   'latitude': 50.037992,
                                   'longitude': 8.582733,
                                   'threshold_distance': 1098},
                    'id': 4992,
                    'identifier': '25L',
                    'localizer': {'beam_width': 4.5,
                                  'frequency': 110700.0,
                                  'heading': 249,
                                  'latitude': 50.026722,
                                  'longitude': 8.53075},
                    'magnetic_heading': 248.0,
                    'start': {'latitude': 50.040053, 'longitude': 8.586531},
                    'strip': {'id': 2496,
                              'length': 13123,
                              'surface': 'CON',
                              'width': 147}},
            turnoff=8998.2717013888887)])
        heading_track = load(os.path.join(test_data_path, 'HeadingDeviationFromRunway_heading_track.nod'))
        to_runway = load(os.path.join(test_data_path, 'HeadingDeviationFromRunway_runway.nod'))
        takeoff = load(os.path.join(test_data_path, 'HeadingDeviationFromRunway_takeoff.nod'))

        deviation = TrackDeviationFromRunway()
        deviation.get_derived((heading_track, None, takeoff, to_runway, apps))
        # check average stays close to 0
        self.assertAlmostEqual(np.ma.average(deviation.array[8775:8975]), 1.5, places = 1)
        self.assertAlmostEqual(np.ma.min(deviation.array[8775:8975]), -10.5, places = 1)
        self.assertAlmostEqual(np.ma.max(deviation.array[8775:8975]), 12.3, places = 1)


class TestHeadingIncreasing(unittest.TestCase):
    def test_can_operate(self):
        expected = [('Heading Continuous',)]
        opts = HeadingIncreasing.get_operational_combinations()
        self.assertEqual(opts, expected)
        
    def test_heading_increasing(self):
        head = P('Heading Continuous', array=np.ma.array([0.0,1.0,-2.0]),
                 frequency=0.5)
        head_inc=HeadingIncreasing()
        head_inc.derive(head)
        expected = np.ma.array([0.0, 1.0, 5.0])
        ma_test.assert_array_equal(head_inc.array, expected)
        
        
class TestLatitudeAndLongitudePrepared(unittest.TestCase):
    def test_can_operate(self):
        combinations = LatitudePrepared.get_operational_combinations()
        # Longitude should be the same list
        self.assertEqual(combinations, LongitudePrepared.get_operational_combinations())
        # only lat long
        self.assertTrue(('Latitude','Longitude') in combinations)
        # with lat long and all the rest
        self.assertTrue(('Latitude',
                         'Longitude',
                         'Heading True',
                         'Airspeed True',
                         'Latitude At Liftoff',
                         'Longitude At Liftoff',
                         'Latitude At Touchdown',
                         'Longitude At Touchdown') in combinations)
        
        # without lat long
        self.assertTrue(('Heading True',
                         'Airspeed True',
                         'Latitude At Liftoff',
                         'Longitude At Liftoff',
                         'Latitude At Touchdown',
                         'Longitude At Touchdown') in combinations)
        
    def test_latitude_smoothing_basic(self):
        lat = P('Latitude',np.ma.array([0,0,1,2,1,0,0],dtype=float))
        lon = P('Longitude', np.ma.array([0,0,0,0,0,0,0.001],dtype=float))
        smoother = LatitudePrepared()
        smoother.get_derived([lat,lon])
        # An output warning of smooth cost function closing with cost > 1 is
        # normal and arises because the data sample is short.
        expected = [0.0, 0.0, 0.00088, 0.00088, 0.00088, 0.0, 0.0]
        np.testing.assert_almost_equal(smoother.array, expected, decimal=5)

    def test_latitude_smoothing_masks_static_data(self):
        lat = P('Latitude',np.ma.array([0,0,1,2,1,0,0],dtype=float))
        lon = P('Longitude', np.ma.zeros(7,dtype=float))
        smoother = LatitudePrepared()
        smoother.get_derived([lat,lon])
        self.assertEqual(np.ma.count(smoother.array),0) # No non-masked values.
        
    def test_latitude_smoothing_short_array(self):
        lat = P('Latitude',np.ma.array([0,0],dtype=float))
        lon = P('Longitude', np.ma.zeros(2,dtype=float))
        smoother = LatitudePrepared()
        smoother.get_derived([lat,lon])
        
    def test_longitude_smoothing_basic(self):
        lat = P('Latitude',np.ma.array([0,0,1,2,1,0,0],dtype=float))
        lon = P('Longitude', np.ma.array([0,0,-2,-4,-2,0,0],dtype=float))
        smoother = LongitudePrepared()
        smoother.get_derived([lat,lon])
        # An output warning of smooth cost function closing with cost > 1 is
        # normal and arises because the data sample is short.
        expected = [0.0, 0.0, -0.00176, -0.00176, -0.00176, 0.0, 0.0]
        np.testing.assert_almost_equal(smoother.array, expected, decimal=5)


class TestHeading(unittest.TestCase):
    def test_can_operate(self):
        self.assertEqual(Heading.get_operational_combinations(),
            [('Heading True Continuous', 'Magnetic Variation')])
        
    def test_basic(self):
        true = P('Heading True Continuous', np.ma.array([0,5,6,355,356]))
        var = P('Magnetic Variation',np.ma.array([2,3,-8,-7,9]))
        head = Heading()
        head.derive(true, var)
        expected = P('Heading True', np.ma.array([358.0, 2.0, 14.0, 2.0, 347.0]))
        ma_test.assert_array_equal(head.array, expected.array)


class TestHeadingTrue(unittest.TestCase):
    def test_can_operate(self):
        self.assertEqual(HeadingTrue.get_operational_combinations(),
            [('Heading Continuous', 'Magnetic Variation From Runway'),
             ('Heading Continuous', 'Magnetic Variation'),
             ('Heading Continuous', 'Magnetic Variation From Runway', 'Magnetic Variation')])
        
    def test_basic_magnetic(self):
        head = P('Heading Continuous', np.ma.array([0,5,6,355,356]))
        var = P('Magnetic Variation',np.ma.array([2,3,-8,-7,9]))
        true = HeadingTrue()
        true.derive(head, None, var)
        expected = P('Heading True', np.ma.array([2.0, 8.0, 358.0, 348.0, 5.0]))
        ma_test.assert_array_equal(true.array, expected.array)
        
    def test_from_runway_used_in_preference(self):
        head = P('Heading Continuous', np.ma.array([0,5,6,355,356]))
        mag_var = P('Magnetic Variation',np.ma.array([2,3,-8,-7,9]))
        rwy_var = P('Magnetic Variation From Runway',np.ma.array([0,1,2,3,4]))
        true = HeadingTrue()
        true.derive(head, rwy_var, mag_var)
        expected = P('Heading True', np.ma.array([0, 6, 8, 358, 0]))
        ma_test.assert_array_equal(true.array, expected.array)


class TestILSFrequency(unittest.TestCase):
    def test_can_operate(self):
        expected = [('ILS (1) Frequency', 'ILS (2) Frequency',),
                    ('ILS-VOR (1) Frequency', 'ILS-VOR (2) Frequency',),
                    ('ILS (1) Frequency', 'ILS (2) Frequency',
                     'ILS-VOR (1) Frequency', 'ILS-VOR (2) Frequency',)]
        opts = ILSFrequency.get_operational_combinations()
        self.assertTrue([e in opts for e in expected])
        
    def test_ils_vor_frequency_in_range(self):
        f1 = P('ILS-VOR (1) Frequency', 
               np.ma.array([1,2,108.10,108.15,111.95,112.00]),
               offset = 0.1, frequency = 0.5)
        f2 = P('ILS-VOR (2) Frequency', 
               np.ma.array([1,2,108.10,108.15,111.95,112.00]),
               offset = 1.1, frequency = 0.5)
        ils = ILSFrequency()
        result = ils.get_derived([None, None, f1, f2])
        expected_array = np.ma.array(
            data=[1,2,108.10,108.15,111.95,112.00], 
             mask=[True,True,False,False,False,True])
        ma_test.assert_masked_array_approx_equal(result.array, expected_array)
        
    def test_single_ils_vor_frequency_in_range(self):
        f1 = P('ILS-VOR (1) Frequency', 
               np.ma.array(data=[1,2,108.10,108.15,111.95,112.00],
                           mask=[True,True,False,False,False,True]),
               offset = 0.1, frequency = 0.5)
        ils = ILSFrequency()
        result = ils.get_derived([None, None, f1, None])
        expected_array = np.ma.array(
            data=[1,2,108.10,108.15,111.95,112.00], 
             mask=[True,True,False,False,False,True])
        ma_test.assert_masked_array_approx_equal(result.array, expected_array)
        
    def test_ils_frequency_in_range(self):
        f1 = P('ILS (1) Frequency', 
               np.ma.array([1,2,108.10,108.15,111.95,112.00]),
               offset = 0.1, frequency = 0.5)
        f2 = P('ILS (2) Frequency', 
               np.ma.array([1,2,108.10,108.15,111.95,112.00]),
               offset = 1.1, frequency = 0.5)
        ils = ILSFrequency()
        result = ils.get_derived([f1, f2, None, None])
        expected_array = np.ma.array(
            data=[1,2,108.10,108.15,111.95,112.00], 
             mask=[True,True,False,False,False,True])
        ma_test.assert_masked_array_approx_equal(result.array, expected_array)
        
    def test_ils_frequency_matched(self):
        f1 = P('ILS-VOR (1) Frequency', 
               np.ma.array([108.10]*3+[111.95]*3),
               offset = 0.1, frequency = 0.5)
        f2 = P('ILS-VOR (2) Frequency', 
               np.ma.array([108.10,111.95]*3),
               offset = 1.1, frequency = 0.5)
        ils = ILSFrequency()
        result = ils.get_derived([f1, f2])
        expected_array = np.ma.array(
            data= [  99,   99, 108.10, 111.95,   99, 111.95], 
             mask=[True, True,  False,  False, True,  False])
        ma_test.assert_masked_array_approx_equal(result.array, expected_array)

    def test_ils_frequency_different_sample_rates(self):
        f1 = P('ILS-VOR (1) Frequency', 
               np.ma.array([108.10]*3+[111.95]*3),
               frequency = 0.5,
               offset = 0.423828125)
        f2 = P('ILS-VOR (2) Frequency', 
               np.ma.array([108.10]*3),
               frequency = 0.25,
               offset = 1.423828125)
        ils = ILSFrequency()
        result = ils.get_derived([f1, f2])
        expected_array = np.ma.array(
            data= [108.10, 108.10, 108.10, 111.95,   99, 111.95], 
             mask=[  True,  False,  False,   True, True,   True])
        ma_test.assert_masked_array_approx_equal(result.array, expected_array)


class TestILSLocalizerRange(unittest.TestCase):
    @unittest.skip('Test Not Implemented')
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')
        
    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestPitch(unittest.TestCase):
    def test_can_operate(self):
        expected = [('Pitch (1)', 'Pitch (2)')]
        opts = Pitch.get_operational_combinations()
        self.assertEqual(opts, expected)
        
    def test_pitch_combination(self):
        pch = Pitch()
        pch.derive(P('Pitch (1)', np.ma.array(range(5),dtype=float), 1,0.1),
                   P('Pitch (2)', np.ma.array(range(5),dtype=float)+10, 1,0.6)
                  )
        answer = np.ma.array(data=([5.0,5.25,5.75,6.25,6.75,7.25,7.75,8.25,8.75,9.0]))
        combo = P('Pitch',answer,frequency=2,offset=0.1)
        ma_test.assert_array_equal(pch.array, combo.array)
        self.assertEqual(pch.frequency, combo.frequency)
        self.assertEqual(pch.offset, combo.offset)

    def test_pitch_reverse_combination(self):
        pch = Pitch()
        pch.derive(P('Pitch (1)', np.ma.array(range(5),dtype=float)+1, 1,0.95),
                   P('Pitch (2)', np.ma.array(range(5),dtype=float)+10, 1,0.45)
                  )
        answer = np.ma.array(data=(range(10)),mask=([1]+[0]*9))/2.0+5.0
        np.testing.assert_array_equal(pch.array, answer.data)

    def test_pitch_error_different_rates(self):
        pch = Pitch()
        self.assertRaises(AssertionError, pch.derive,
                          P('Pitch (1)', np.ma.array(range(5),dtype=float), 2,0.1),
                          P('Pitch (2)', np.ma.array(range(10),dtype=float)+10, 4,0.6))
        
    def test_pitch_different_offsets(self):
        pch = Pitch()
        pch.derive(P('Pitch (1)', np.ma.array(range(5),dtype=float), 1,0.11),
                   P('Pitch (2)', np.ma.array(range(5),dtype=float), 1,0.6))
        # This originally produced an error, but with amended merge processes
        # this is not necessary. Simply check the result is the right length.
        self.assertEqual(len(pch.array),10)
        

class TestVerticalSpeed(unittest.TestCase):
    def test_can_operate(self):
        self.assertEqual(VerticalSpeed.get_operational_combinations(),
                         [('Altitude STD Smoothed',),
                           ('Altitude STD Smoothed', 'Frame')])
                         
    def test_vertical_speed_basic(self):
        alt_std = P('Altitude STD Smoothed', np.ma.array([100]*10))
        vert_spd = VerticalSpeed()
        vert_spd.derive(alt_std, None)
        expected = np.ma.array(data=[0]*10, dtype=np.float,
                             mask=False)
        ma_test.assert_masked_array_approx_equal(vert_spd.array, expected)
    
    def test_vertical_speed_alt_std_only(self):
        alt_std = P('Altitude STD Smoothed', np.ma.arange(100, 200, 10))
        vert_spd = VerticalSpeed()
        vert_spd.derive(alt_std, None)
        expected = np.ma.array(data=[600] * 10, dtype=np.float,
                               mask=False) #  10 ft/sec = 600 fpm
        ma_test.assert_masked_array_approx_equal(vert_spd.array, expected)


class TestVerticalSpeedForFlightPhases(unittest.TestCase):
    def test_can_operate(self):
        expected = [('Altitude STD Smoothed',)]
        opts = VerticalSpeedForFlightPhases.get_operational_combinations()
        self.assertEqual(opts, expected)
        
    def test_vertical_speed_for_flight_phases_basic(self):
        alt_std = P('Altitude STD Smoothed', np.ma.arange(10))
        vert_spd = VerticalSpeedForFlightPhases()
        vert_spd.derive(alt_std)
        expected = np.ma.array(data=[60]*10, dtype=np.float, mask=False)
        np.testing.assert_array_equal(vert_spd.array, expected)

    def test_vertical_speed_for_flight_phases_level_flight(self):
        alt_std = P('Altitude STD Smoothed', np.ma.array([100]*10))
        vert_spd = VerticalSpeedForFlightPhases()
        vert_spd.derive(alt_std)
        expected = np.ma.array(data=[0]*10, dtype=np.float, mask=False)
        np.testing.assert_array_equal(vert_spd.array, expected)

        
class TestRateOfTurn(unittest.TestCase):
    def test_can_operate(self):
        expected = [('Heading Continuous',)]
        opts = RateOfTurn.get_operational_combinations()
        self.assertEqual(opts, expected)
       
    def test_rate_of_turn(self):
        rot = RateOfTurn()
        rot.derive(P('Heading Continuous', np.ma.array(range(10))))
        answer = np.ma.array(data=[1]*10, dtype=np.float)
        np.testing.assert_array_equal(rot.array, answer) # Tests data only; NOT mask
       
    def test_rate_of_turn_phase_stability(self):
        rot = RateOfTurn()
        rot.derive(P('Heading Continuous', np.ma.array([0,0,2,4,2,0,0],
                                                          dtype=float)))
        answer = np.ma.array([0,1.95,0.5,0,-0.5,-1.95,0])
        ma_test.assert_masked_array_approx_equal(rot.array, answer)
        
    def test_sample_long_gentle_turn(self):
        # Sample taken from a long circling hold pattern
        head_cont = P(array=np.ma.array(
            np.load(os.path.join(test_data_path, 'heading_continuous_in_hold.npy'))), frequency=2)
        rot = RateOfTurn()
        rot.get_derived((head_cont,))
        np.testing.assert_allclose(rot.array[50:1150],
                                   np.ones(1100, dtype=float)*2.1, rtol=0.1)
        
        
class TestMach(unittest.TestCase):
    def test_can_operate(self):
        opts = Mach.get_operational_combinations()
        self.assertEqual(opts, [('Airspeed', 'Altitude STD')])
        
    def test_all_cases(self):
        cas = P('Airspeed', np.ma.array(data=[0, 100, 200, 200, 200, 500, 200],
                                        mask=[0,0,0,0,1,0,0], dtype=float))
        alt = P('Altitude STD', np.ma.array(data=[0, 10000, 20000, 30000, 30000, 45000, 20000],
                                        mask=[0,0,0,0,0,0,1], dtype=float))
        mach = Mach()
        mach.derive(cas, alt)
        expected = np.ma.array(data=[0, 0.182, 0.4402, 0.5407, 0.5407, 1.6825, 45000],
                                        mask=[0,0,0,0,1,1,1], dtype=float)
        ma_test.assert_masked_array_approx_equal(mach.array, expected, decimal=2)


class TestHeadwind(unittest.TestCase):
    def test_can_operate(self):
        opts = Headwind.get_operational_combinations()
        self.assertEqual(opts, [
            ('Wind Speed', 'Wind Direction Continuous', 'Heading True Continuous', 'Altitude AAL'),
            ('Airspeed True', 'Wind Speed', 'Wind Direction Continuous', 'Heading True Continuous', 'Altitude AAL'),
            ('Wind Speed', 'Wind Direction Continuous', 'Heading True Continuous', 'Altitude AAL', 'Groundspeed'),
            ('Airspeed True', 'Wind Speed', 'Wind Direction Continuous', 'Heading True Continuous', 'Altitude AAL', 'Groundspeed'),
        ])            
    
    def test_real_example(self):
        ws = P('Wind Speed', np.ma.array([84.0]))
        wd = P('Wind Direction Continuous', np.ma.array([-21]))
        head=P('Heading True Continuous', np.ma.array([30]))
        hw = Headwind()
        hw.derive(None, ws, wd, head, None, None)
        expected = np.ma.array([52.8629128481863])
        self.assertAlmostEqual(hw.array.data, expected.data)
        
    def test_odd_angles(self):
        ws = P('Wind Speed', np.ma.array([20.0]*8))
        wd = P('Wind Direction Continuous', np.ma.array([0, 90, 180, -180, -90, 360, 23, -23], dtype=float))
        head=P('Heading True Continuous', np.ma.array([-180, -90, 0, 180, 270, 360*15, 361*23, 359*23], dtype=float))
        hw = Headwind()
        hw.derive(None, ws, wd, head, None, None)
        expected = np.ma.array([-20]*3+[20]*5)
        ma_test.assert_masked_array_almost_equal (hw.array, expected)

    def test_headwind_below_100ft(self):
        # create consistent 20 kt windspeed on the tail
        wspd = P('Wind Speed', np.ma.array([20]*20))
        wdir = P('Wind Direction Continuous', np.ma.array([180]*20))
        head = P('Heading True Continuous', np.ma.array([0]*20))
        # create a 40 kt difference between Airspeed and speed over ground
        gspd = P('Groundspeed', np.ma.array([220]*20))
        aspd = P('Airpseed True', np.ma.array([180]*20))
        # first 5 and last 5 samples are below 100ft
        alt = P('Altitude AAL', np.ma.array([50]*5 + [5000]*10 + [40]*5))
        hw = Headwind()
        hw.derive(aspd, wspd, wdir, head, alt, gspd)
        # below 100ft
        np.testing.assert_equal(hw.array[:5], [-40]*5)
        # above 100ft
        np.testing.assert_equal(hw.array[5:15], [-20]*10)
        # below 100ft
        np.testing.assert_equal(hw.array[15:], [-40]*5)


class TestWindAcrossLandingRunway(unittest.TestCase):
    def test_can_operate(self):
        opts = WindAcrossLandingRunway.get_operational_combinations()
        expected = [('Wind Speed', 'Wind Direction True Continuous', 'FDR Landing Runway'),
                    ('Wind Speed', 'Wind Direction Continuous', 'Heading During Landing'),
                    ('Wind Speed', 'Wind Direction True Continuous', 'Wind Direction Continuous', 'FDR Landing Runway'),
                    ('Wind Speed', 'Wind Direction True Continuous', 'Wind Direction Continuous', 'Heading During Landing'),
                    ('Wind Speed', 'Wind Direction True Continuous', 'FDR Landing Runway', 'Heading During Landing'),
                    ('Wind Speed', 'Wind Direction Continuous', 'FDR Landing Runway', 'Heading During Landing'),
                    ('Wind Speed', 'Wind Direction True Continuous', 'Wind Direction Continuous', 'FDR Landing Runway', 'Heading During Landing')]
        self.assertEqual(opts, expected)
    
    def test_real_example(self):
        ws = P('Wind Speed', np.ma.array([84.0]))
        wd = P('Wind Direction Continuous', np.ma.array([-21]))
        land_rwy = A('FDR Landing Runway')
        land_rwy.value = {'start': {'latitude': 60.18499999999998,
                                    'longitude': 11.073744}, 
                          'end': {'latitude': 60.216066999999995,
                                  'longitude': 11.091663999999993}}
        
        walr = WindAcrossLandingRunway()
        walr.derive(ws,wd,None,land_rwy,None)
        expected = np.ma.array([50.55619778])
        self.assertAlmostEqual(walr.array.data, expected.data, 1)
        
    def test_error_cases(self):
        ws = P('Wind Speed', np.ma.array([84.0]))
        wd = P('Wind Direction True Continuous', np.ma.array([-21]))
        land_rwy = A('FDR Landing Runway')
        land_rwy.value = {}
        walr = WindAcrossLandingRunway()

        walr.derive(ws,wd,None,land_rwy,None)
        self.assertEqual(len(walr.array.data), len(ws.array.data))
        self.assertEqual(walr.array.data[0],0.0)
        self.assertEqual(walr.array.mask[0],1)
        
        walr.derive(ws,wd,None)
        self.assertEqual(len(walr.array.data), len(ws.array.data))
        self.assertEqual(walr.array.data[0],0.0)
        self.assertEqual(walr.array.mask[0],1)


class TestAOA(unittest.TestCase):
    def test_can_operate(self):
        opts = AOA.get_operational_combinations()
        self.assertEqual(opts, [
            ('AOA (L)',),
            ('AOA (R)',),
            ('AOA (L)', 'AOA (R)')])
        
    def test_derive(self):
        aoa_l = P('AOA (L)', [4.921875, 4.5703125, 4.5703125, 4.5703125,
                              4.570315, 4.5703125, 4.5703125, 4.9213875],
                  frequency=1.0, offset=0.1484375)
        
        aoa_r = P('AOA (R)', [4.881875, 4.5703125, 4.5712125, 4.544125],
                          frequency=0.5, offset=0.6484375)
        aoa = AOA()
        res = aoa.derive(aoa_l, aoa_r)
        self.assertEqual(aoa.hz, 2)
        self.assertEqual(aoa.offset, 0)
        
    def test_Derive_only_left(self):
        aoa_l = P('AOA (L)', [4.921875, 4.5703125, 4.5703125, 4.5703125,
                              4.570315, 4.5703125, 4.5703125, 4.9213875],
                  frequency=1.0, offset=0.1484375)
        
        aoa = AOA()
        res = aoa.derive(aoa_l, None)
        self.assertEqual(aoa.hz, 1)
        self.assertEqual(aoa.offset, 0.1484375)
        

class TestAccelerationNormalOffsetRemoved(unittest.TestCase):
    @unittest.skip('Test Not Implemented')
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')
        
    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')

# Also Accelerations Lateral and Longitudinal are not tested yet.

class TestAileron(unittest.TestCase):
    
    def test_can_operate(self):
        opts = Aileron.get_operational_combinations()
        self.assertTrue(opts,
                        [('Aileron (L)',),
                         ('Aileron (R)',),
                         ('Aileron (L)', 'Aileron (R)'),
                        ])

    def test_normal_two_sensors(self):
        left = P('Aileron (L)', np.ma.array([1.0]*2+[2.0]*2), frequency=0.5, offset=0.1)
        right = P('Aileron (R)', np.ma.array([2.0]*2+[1.0]*2), frequency=0.5, offset=1.1)
        aileron = Aileron()
        aileron.get_derived([left, right])
        expected_data = np.ma.array([0.0, 1.5, 1.75, 1.5])
        expected_data[0] = np.ma.masked
        np.testing.assert_array_equal(aileron.array, expected_data)
        self.assertEqual(aileron.frequency, 0.5)
        self.assertEqual(aileron.offset, 0.1)

    def test_left_only(self):
        left = P('Aileron (L)', np.ma.array([1.0]*2+[2.0]*2), frequency=0.5, offset=0.1)
        aileron = Aileron()
        aileron.get_derived([left, None])
        expected_data = left.array
        np.testing.assert_array_equal(aileron.array, expected_data)
        self.assertEqual(aileron.frequency, 0.5)
        self.assertEqual(aileron.offset, 0.1)

    def test_right_only(self):
        right = P('Aileron (R)', np.ma.array([3.0]*2+[2.0]*2), frequency=2.0, offset = 0.3)
        aileron = Aileron()
        aileron.get_derived([None, right])
        expected_data = right.array
        np.testing.assert_array_equal(aileron.array, expected_data)
        self.assertEqual(aileron.frequency, 2.0)
        self.assertEqual(aileron.offset, 0.3)    
        
    def test_aileron_with_flaperon(self):
        al = load(os.path.join(test_data_path, 'aileron_left.nod'))
        ar = load(os.path.join(test_data_path, 'aileron_right.nod'))
        ail = Aileron()
        ail.derive(al, ar)
        # this section is averaging 4.833 degrees on the way in
        self.assertAlmostEqual(np.ma.average(ail.array[160:600]), 0.04, 1)
        # this section is averaging 9.106 degrees, ensure it gets moved to 0
        #self.assertAlmostEqual(np.ma.average(ail.array[800:1000]), 0.2, 1)
        assert_array_within_tolerance(ail.array[800:1000], 0, 4, 90)


class TestAileronTrim(unittest.TestCase):
    @unittest.skip('Test Not Implemented')
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')
        
    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestAltitudeSTD(unittest.TestCase):
    @unittest.skip('Test Not Implemented')
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')
        
    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestElevator(unittest.TestCase):
    def test_can_operate(self):
        self.assertEqual(Elevator.get_operational_combinations(),
                         [('Elevator (L)',),
                          ('Elevator (R)',),
                          ('Elevator (L)', 'Elevator (R)'),
                          ])
        
    def test_normal_two_sensors(self):
        left = P('Elevator (L)', np.ma.array([1.0]*2+[2.0]*2), frequency=0.5, offset = 0.1)
        right = P('Elevator (R)', np.ma.array([2.0]*2+[1.0]*2), frequency=0.5, offset = 1.1)
        elevator = Elevator()
        elevator.derive(left, right)
        expected_data = np.ma.array([1.5]*3+[1.75]*2+[1.5]*3)
        np.testing.assert_array_equal(elevator.array, expected_data)
        self.assertEqual(elevator.frequency, 1.0)
        self.assertEqual(elevator.offset, 0.1)

    def test_left_only(self):
        left = P('Elevator (L)', np.ma.array([1.0]*2+[2.0]*2), frequency=0.5, offset = 0.1)
        elevator = Elevator()
        elevator.derive(left, None)
        expected_data = left.array
        np.testing.assert_array_equal(elevator.array, expected_data)
        self.assertEqual(elevator.frequency, 0.5)
        self.assertEqual(elevator.offset, 0.1)
    
    def test_right_only(self):
        right = P('Elevator (R)', np.ma.array([3.0]*2+[2.0]*2), frequency=2.0, offset = 0.3)
        elevator = Elevator()
        elevator.derive(None, right)
        expected_data = right.array
        np.testing.assert_array_equal(elevator.array, expected_data)
        self.assertEqual(elevator.frequency, 2.0)
        self.assertEqual(elevator.offset, 0.3)

class TestElevatorLeft(unittest.TestCase):
    def test_can_operate(self):
        opts = ElevatorLeft.get_operational_combinations()
        self.assertEqual(opts, [('Elevator (L) Potentiometer',),
                                ('Elevator (L) Synchro',),
                                ('Elevator (L) Potentiometer','Elevator (L) Synchro'),
                                ])
        
    def test_synchro(self):
        syn=P('Elevator (L) Synchro', np.ma.array(data=[1,2,3,4],
                                                  mask=[0,0,1,0]))
        elevator=ElevatorLeft()
        elevator.derive(None, syn)
        ma_test.assert_array_equal(elevator.array, syn.array)
              
    def test_pot(self):
        pot=P('Elevator (L) Potentiometer', np.ma.array(data=[5,6,7,8],
                                                  mask=[0,1,0,0]))
        elevator=ElevatorLeft()
        elevator.derive(pot, None)
        ma_test.assert_array_equal(elevator.array, pot.array)
              
    def test_both_prefer_syn(self):
        syn=P('Elevator (L) Synchro', np.ma.array(data=[1,2,3,4],
                                                  mask=[0,0,1,0]))
        pot=P('Elevator (L) Potentiometer', np.ma.array(data=[5,6,7,8],
                                                  mask=[0,1,1,0]))
        elevator=ElevatorLeft()
        elevator.derive(pot, syn)
        ma_test.assert_array_equal(elevator.array, syn.array)
              
    def test_both_prefer_pot(self):
        syn=P('Elevator (L) Synchro', np.ma.array(data=[1,2,3,4],
                                                  mask=[1,0,1,0]))
        pot=P('Elevator (L) Potentiometer', np.ma.array(data=[5,6,7,8],
                                                  mask=[0,0,1,0]))
        elevator=ElevatorLeft()
        elevator.derive(pot, syn)
        ma_test.assert_array_equal(elevator.array, pot.array)
              
    def test_both_equally_good(self):
        # Where there is no advantage, adopt the synchro which should be a better transducer.
        syn=P('Elevator (L) Synchro', np.ma.array(data=[1,2,3,4],
                                                  mask=[0,0,0,0]))
        pot=P('Elevator (L) Potentiometer', np.ma.array(data=[5,6,7,8],
                                                  mask=[0,0,0,0]))
        elevator=ElevatorLeft()
        elevator.derive(pot, syn)
        ma_test.assert_array_equal(elevator.array, syn.array)
              
class TestElevatorRight(unittest.TestCase):
    def test_can_operate(self):
        opts = ElevatorRight.get_operational_combinations()
        self.assertEqual(opts, [('Elevator (R) Potentiometer',),
                                ('Elevator (R) Synchro',),
                                ('Elevator (R) Potentiometer','Elevator (R) Synchro'),
                                ])
        
    def test_synchro(self):
        syn=P('Elevator (L) Synchro', np.ma.array(data=[1,2,3,4],
                                                  mask=[0,0,1,0]))
        elevator=ElevatorRight()
        elevator.derive(None, syn)
        ma_test.assert_array_equal(elevator.array, syn.array)
              
    def test_pot(self):
        pot=P('Elevator (L) Potentiometer', np.ma.array(data=[5,6,7,8],
                                                  mask=[0,1,0,0]))
        elevator=ElevatorRight()
        elevator.derive(pot, None)
        ma_test.assert_array_equal(elevator.array, pot.array)
              
    def test_both_prefer_syn(self):
        syn=P('Elevator (L) Synchro', np.ma.array(data=[1,2,3,4],
                                                  mask=[0,0,1,0]))
        pot=P('Elevator (L) Potentiometer', np.ma.array(data=[5,6,7,8],
                                                  mask=[0,1,1,0]))
        elevator=ElevatorRight()
        elevator.derive(pot, syn)
        ma_test.assert_array_equal(elevator.array, syn.array)
              
    def test_both_prefer_pot(self):
        syn=P('Elevator (L) Synchro', np.ma.array(data=[1,2,3,4],
                                                  mask=[1,0,1,0]))
        pot=P('Elevator (L) Potentiometer', np.ma.array(data=[5,6,7,8],
                                                  mask=[0,0,1,0]))
        elevator=ElevatorRight()
        elevator.derive(pot, syn)
        ma_test.assert_array_equal(elevator.array, pot.array)
              
    def test_both_equally_good(self):
        # Where there is no advantage, adopt the synchro which should be a better transducer.
        syn=P('Elevator (L) Synchro', np.ma.array(data=[1,2,3,4],
                                                  mask=[0,0,0,0]))
        pot=P('Elevator (L) Potentiometer', np.ma.array(data=[5,6,7,8],
                                                  mask=[0,0,0,0]))
        elevator=ElevatorRight()
        elevator.derive(pot, syn)
        ma_test.assert_array_equal(elevator.array, syn.array)


class TestEng_FuelFlow(unittest.TestCase):
    def test_can_operate(self):
        for n in range(1, 5):
            self.assertTrue(Eng_FuelFlow.can_operate(
                ('Eng (%d) Fuel Flow' % n,)))

    def test_derive(self):
        ff1_arr = np.ma.arange(100)
        ff2_arr = np.ma.arange(100)[::-1]
        ff1 = P(name='Eng (1) Fuel Flow', array=ff1_arr, frequency=1)
        ff2 = P(name='Eng (2) Fuel Flow', array=ff2_arr, frequency=1)

        expected_arr = np.ma.array([99] * 100)

        node = Eng_FuelFlow()
        node.derive(ff1, ff2, None, None)

        np.testing.assert_array_equal(node.array, expected_arr)


class TestEng_FuelFlowMax(unittest.TestCase):
    def test_can_operate(self):
        for n in range(1, 5):
            self.assertTrue(Eng_FuelFlowMax.can_operate(
                ('Eng (%d) Fuel Flow' % n,)))

    def test_derive(self):
        ff1_arr = np.ma.arange(100)
        ff2_arr = np.ma.arange(100)[::-1]
        ff1 = P(name='Eng (1) Fuel Flow', array=ff1_arr, frequency=1)
        ff2 = P(name='Eng (2) Fuel Flow', array=ff2_arr, frequency=1)

        expected_arr = np.ma.concatenate([np.ma.arange(99, 49, -1),
                                          np.ma.arange(50, 100)])

        node = Eng_FuelFlowMax()
        node.derive(ff1, ff2, None, None)

        np.testing.assert_array_equal(node.array, expected_arr)


class TestEng_FuelFlowMin(unittest.TestCase):
    def test_can_operate(self):
        for n in range(1, 5):
            self.assertTrue(Eng_FuelFlowMin.can_operate(
                ('Eng (%d) Fuel Flow' % n,)))

    def test_derive(self):
        ff1_arr = np.ma.arange(100)
        ff2_arr = np.ma.arange(100)[::-1]
        ff1 = P(name='Eng (1) Fuel Flow', array=ff1_arr, frequency=1)
        ff2 = P(name='Eng (2) Fuel Flow', array=ff2_arr, frequency=1)

        expected_arr = np.ma.concatenate([np.ma.arange(50),
                                          np.ma.arange(49, -1, -1)])

        node = Eng_FuelFlowMin()
        node.derive(ff1, ff2, None, None)

        np.testing.assert_array_equal(node.array, expected_arr)


class TestEng_1_FuelBurn(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = Eng_1_FuelBurn
        self.operational_combinations = [('Eng (1) Fuel Flow', )]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestEng_2_FuelBurn(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = Eng_2_FuelBurn
        self.operational_combinations = [('Eng (2) Fuel Flow', )]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestEng_3_FuelBurn(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = Eng_3_FuelBurn
        self.operational_combinations = [('Eng (3) Fuel Flow', )]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestEng_4_FuelBurn(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = Eng_4_FuelBurn
        self.operational_combinations = [('Eng (4) Fuel Flow', )]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestEng_FuelBurn(unittest.TestCase):

    @unittest.skip('Test Not Implemented')
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestEng_GasTempAvg(unittest.TestCase):
    @unittest.skip('Test Not Implemented')
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')
        
    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestEng_GasTempMax(unittest.TestCase):
    @unittest.skip('Test Not Implemented')
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')
        
    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestEng_GasTempMin(unittest.TestCase):
    @unittest.skip('Test Not Implemented')
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')
        
    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestEng_OilPressAvg(unittest.TestCase):
    @unittest.skip('Test Not Implemented')
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')
        
    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestEng_OilPressMax(unittest.TestCase):
    @unittest.skip('Test Not Implemented')
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')
        
    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestEng_OilPressMin(unittest.TestCase):
    @unittest.skip('Test Not Implemented')
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')
        
    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestEng_OilQtyAvg(unittest.TestCase):
    @unittest.skip('Test Not Implemented')
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')
        
    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestEng_OilQtyMax(unittest.TestCase):
    @unittest.skip('Test Not Implemented')
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')
        
    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestEng_OilQtyMin(unittest.TestCase):
    @unittest.skip('Test Not Implemented')
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')
        
    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestEng_OilTempAvg(unittest.TestCase):
    @unittest.skip('Test Not Implemented')
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')
        
    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestEng_OilTempMax(unittest.TestCase):
    @unittest.skip('Test Not Implemented')
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')
        
    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestEng_OilTempMin(unittest.TestCase):
    @unittest.skip('Test Not Implemented')
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')
        
    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestEng_TorqueAvg(unittest.TestCase):
    @unittest.skip('Test Not Implemented')
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')
        
    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestEng_TorqueMax(unittest.TestCase):
    @unittest.skip('Test Not Implemented')
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')
        
    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestEng_TorqueMin(unittest.TestCase):
    @unittest.skip('Test Not Implemented')
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')
        
    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestEng_TorquePercentAvg(unittest.TestCase):
    
    def setUp(self):
        self.node_class = Eng_TorquePercentAvg

    def test_can_operate(self):
        poss_combs = self.node_class.get_operational_combinations()
        self.assertTrue(('Eng (1) Torque [%]',) in poss_combs)
        self.assertTrue(('Eng (2) Torque [%]',) in poss_combs)
        self.assertTrue(('Eng (3) Torque [%]',) in poss_combs)
        self.assertTrue(('Eng (4) Torque [%]',) in poss_combs)

    def test_derive(self):
        eng_1_array =    [0, 30, 50, 80, 100,   100, 70, 70, 70, 50, 50, 10,  0,  0, 0]
        eng_3_array =    [0,  0, 30, 60,  85,   100, 70, 70, 70, 50, 50, 30, 10, 10, 0]
        expected_array = [0, 15, 40, 70,  92.5, 100, 70, 70, 70, 50, 50, 20,  5,  5, 0]

        eng_1 = P(name='Eng (1) Torque [%]', array=eng_1_array, frequency=1,
                 offset=0.1)

        eng_3 = P(name='Eng (3) Torque [%]', array=eng_3_array, frequency=1,
                         offset=0.5)

        node = self.node_class()
        node.derive(eng_1, None, eng_3, None)

        np.testing.assert_array_equal(node.array, expected_array)
        self.assertEqual(node.offset, 0.3)


class TestEng_TorquePercentMax(unittest.TestCase):

    def setUp(self):
        self.node_class = Eng_TorquePercentMax

    def test_can_operate(self):
        poss_combs = self.node_class.get_operational_combinations()
        self.assertTrue(('Eng (1) Torque [%]',) in poss_combs)
        self.assertTrue(('Eng (2) Torque [%]',) in poss_combs)
        self.assertTrue(('Eng (3) Torque [%]',) in poss_combs)
        self.assertTrue(('Eng (4) Torque [%]',) in poss_combs)


    def test_derive(self):
        eng_1_array =    [0, 30, 50, 80, 100, 100, 70, 70, 70, 50, 50, 10,  0,  0, 0]
        eng_3_array =    [0,  0, 30, 60,  85, 100, 70, 70, 70, 50, 50, 30, 10, 10, 0]
        expected_array = [0, 30, 50, 80, 100, 100, 70, 70, 70, 50, 50, 30, 10, 10, 0]

        eng_1 = P(name='Eng (1) Torque [%]', array=eng_1_array, frequency=1,
                 offset=0.1)

        eng_3 = P(name='Eng (3) Torque [%]', array=eng_3_array, frequency=1,
                offset=0.5)

        node = self.node_class()
        node.derive(eng_1, None, eng_3, None)

        np.testing.assert_array_equal(node.array, expected_array)
        self.assertEqual(node.offset, 0.3)


class TestEng_TorquePercentMin(unittest.TestCase):

    def setUp(self):
        self.node_class = Eng_TorquePercentMin

    def test_can_operate(self):
        poss_combs = self.node_class.get_operational_combinations()
        self.assertTrue(('Eng (1) Torque [%]',) in poss_combs)
        self.assertTrue(('Eng (2) Torque [%]',) in poss_combs)
        self.assertTrue(('Eng (3) Torque [%]',) in poss_combs)
        self.assertTrue(('Eng (4) Torque [%]',) in poss_combs)


    def test_derive(self):
        eng_1_array =    [0, 30, 50, 80, 100, 100, 70, 70, 70, 50, 50, 10,  0,  0, 0]
        eng_3_array =    [0,  0, 30, 60,  85, 100, 70, 70, 70, 50, 50, 30, 10, 10, 0]
        expected_array = [0,  0, 30, 60,  85, 100, 70, 70, 70, 50, 50, 10,  0,  0, 0]

        eng_1 = P(name='Eng (1) Torque [%]', array=eng_1_array, frequency=1,
                 offset=0.1)

        eng_3 = P(name='Eng (3) Torque [%]', array=eng_3_array, frequency=1,
                         offset=0.5)

        node = self.node_class()
        node.derive(eng_1, None, eng_3, None)

        np.testing.assert_array_equal(node.array, expected_array)
        self.assertEqual(node.offset, 0.3)


class TestEng_VibBroadbandMax(unittest.TestCase, NodeTest):
    
    def setUp(self):
        self.node_class = Eng_VibBroadbandMax
        self.operational_combinations = [
            ('Eng (1) Vib Broadband',),
            ('Eng (1) Vib Broadband Accel A',),
            ('Eng (1) Vib Broadband Accel B',),
            ('Eng (1) Vib Broadband', 'Eng (2) Vib Broadband', 'Eng (3) Vib Broadband', 'Eng (4) Vib Broadband'),
            ('Eng (1) Vib Broadband Accel A', 'Eng (2) Vib Broadband Accel A', 'Eng (3) Vib Broadband Accel A', 'Eng (4) Vib Broadband Accel A'),
            ('Eng (1) Vib Broadband Accel B', 'Eng (2) Vib Broadband Accel B', 'Eng (3) Vib Broadband Accel B', 'Eng (4) Vib Broadband Accel B',),
        ]
    
    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestEng_VibN1Max(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = Eng_VibN1Max
        self.operational_combination_length = 255
        self.check_operational_combination_length_only = True

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestEng_VibN2Max(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = Eng_VibN2Max
        self.operational_combination_length = 255
        self.check_operational_combination_length_only = True

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestEng_VibN3Max(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = Eng_VibN3Max
        self.operational_combination_length = 15
        self.check_operational_combination_length_only = True

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestEng_VibAMax(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = Eng_VibAMax
        self.operational_combination_length = 15
        self.check_operational_combination_length_only = True

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestEng_VibBMax(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = Eng_VibBMax
        self.operational_combination_length = 15
        self.check_operational_combination_length_only = True

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestEng_VibCMax(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = Eng_VibCMax
        self.operational_combination_length = 15
        self.check_operational_combination_length_only = True

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestEngTPRLimitDifference(unittest.TestCase):
    def test_can_operate(self):
        self.assertEqual(EngTPRLimitDifference.get_operational_combinations(),
                         [('Eng (*) TPR Max', 'Eng TPR Limit Max')])
    
    def test_derive_basic(self):
        eng_tpr_max_array = np.ma.concatenate([
            np.ma.arange(0, 150, 10), np.ma.arange(150, 0, -10)])
        eng_tpr_limit_array = np.ma.concatenate([
            np.ma.arange(10, 110, 10), [110] * 10, np.ma.arange(110, 10, -10)])
        eng_tpr_max = P('Eng (*) TPR Max', array=eng_tpr_max_array)
        eng_tpr_limit = P('Eng (*) TPR Limit Max', array=eng_tpr_limit_array)
        node = EngTPRLimitDifference()
        node.derive(eng_tpr_max, eng_tpr_limit)
        expected = [0] * 5
        self.assertEqual(
            node.array.tolist(),
            [-10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, 0, 10, 20,
             30, 40, 30, 20, 10, 0, -10, -10, -10, -10, -10, -10, -10, -10, -10,
             -10])


class TestFlapAngle(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = FlapAngle
        self.operational_combinations = [
            ('Flap Angle (L)',),
            ('Flap Angle (R)',),
            ('Flap Angle (L) Inboard',),
            ('Flap Angle (R) Inboard',),
            ('Flap Angle (L)', 'Flap Angle (R)'),
            ('Flap Angle (L) Inboard', 'Flap Angle (R) Inboard'),
            ('Flap Angle (L)', 'Flap Angle (R)', 'Flap Angle (C)', 'Flap Angle (MCP)'),
            ('Flap Angle (L)', 'Flap Angle (R)', 'Flap Angle (L) Inboard', 'Flap Angle (R) Inboard', 'Frame'),
            ('Flap Angle (L)', 'Flap Angle (R)', 'Flap Angle (L) Inboard', 'Flap Angle (R) Inboard', 'Slat Angle', 'Frame'),
        ]

    @patch('analysis_engine.derived_parameters.at')
    def test_derive_787(self, at):
        at.get_lever_angles.return_value = {
            0:  (0, 0, None),
            1:  (50, 0, None),
            5:  (50, 5, None),
            15: (50, 15, None),
            20: (50, 20, None),
            25: (100, 20, None),
            30: (100, 30, None),
        }
        _load = lambda x: load(os.path.join(test_data_path, x))
        flap_angle_l = _load('787_flap_angle_l.nod')
        flap_angle_r = _load('787_flap_angle_r.nod')
        slat_angle_l = _load('787_slat_l.nod')
        slat_angle_r = _load('787_slat_r.nod')
        slat_angle = SlatAngle()
        slat_angle.derive(slat_angle_l, slat_angle_r)
        family = A('Family', 'B787')
        node = self.node_class()
        node.derive(flap_angle_l, flap_angle_r, None, None, None, None, slat_angle, None, family)
        # Include transitions:
        self.assertEqual(node.array[18635], 0.70000000000000007)
        self.assertEqual(node.array[18650], 1.0)
        self.assertEqual(node.array[18900], 5.0)
        # The original flap data does not always record exact flap settings:
        self.assertEqual(node.array[19070], 19.945)
        self.assertEqual(node.array[19125], 24.945)
        self.assertEqual(node.array[19125], 24.945)
        self.assertEqual(node.array[19250], 30.0)

    def test__combine_flap_set_basic(self):
        lever_angles = {
            0:  (0, 0, None),
            1:  (50, 0, None),
            5:  (50, 5, None),
            15: (50, 15, None),
            20: (50, 20, None),
            25: (100, 20, None),
            30: (100, 30, None),
        }
        slat_array = np.ma.array((0.0, 50, 50, 50, 50, 100, 100))
        flap_array = np.ma.array((0.0, 0, 5, 15, 20, 20, 30))
        array = self.node_class._combine_flap_slat(slat_array, flap_array, lever_angles)
        self.assertEqual(array.tolist(), [0.0, 1.0, 5.0, 15.0, 20.0, 25.0, 30.0])

    def test_hercules(self):
        parameter = P(array=np.ma.array(range(0, 5000, 100) + range(5000, 0, -200)))
        frame = A('Frame', 'L382-Hercules')
        node = self.node_class()
        node.derive(parameter, None, None, None, None, None, None, None, frame)
        self.assertAlmostEqual(node.array[50], 2500, 1)


class TestHeadingTrueContinuous(unittest.TestCase):
    @unittest.skip('Test Not Implemented')
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')
        
    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestILSGlideslope(unittest.TestCase):
    @unittest.skip('Test Not Implemented')
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')
        
    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestILSLocalizer(unittest.TestCase):
    @unittest.skip('Test Not Implemented')
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')
        
    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestLatitudePrepared(unittest.TestCase):
    @unittest.skip('Test Not Implemented')
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')
        
    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestLatitudeSmoothed(unittest.TestCase):
    def test_can_operate(self):
        combinations = LatitudeSmoothed.get_operational_combinations()
        self.assertTrue(all('Latitude Prepared' in c for c in combinations))

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestLongitudePrepared(unittest.TestCase):
    @unittest.skip('Test Not Implemented')
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')
        
    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestLongitudeSmoothed(unittest.TestCase):
    def test_can_operate(self):
        combinations = LongitudeSmoothed.get_operational_combinations()
        self.assertTrue(all('Longitude Prepared' in c for c in combinations))
        
    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestMagneticVariation(unittest.TestCase):
    def test_can_operate(self):
        combinations = MagneticVariation.get_operational_combinations()
        self.assertTrue(
            ('Latitude', 'Longitude', 'Altitude AAL', 'Start Datetime') in combinations)
        self.assertTrue(
            ('Latitude (Coarse)', 'Longitude (Coarse)', 'Altitude AAL', 'Start Datetime') in combinations)
        self.assertTrue(
            ('Latitude', 'Latitude (Coarse)', 'Longitude', 'Longitude (Coarse)', 'Altitude AAL', 'Start Datetime') in combinations)        
        
    def test_derive(self):
        mag_var = MagneticVariation()
        lat = P('Latitude', array=np.ma.arange(10, 14, 0.01))
        lat.array[3] = np.ma.masked
        lon = P('Longitude', array=np.ma.arange(-10, -14, -0.01))
        lon.array[2:4] = np.ma.masked
        alt_aal = P('Altitude AAL', array=np.ma.arange(20000, 24000, 10))
        alt_aal.array[4] = np.ma.masked
        start_datetime = A('Start Datetime',
                           value=datetime.datetime(2013, 3, 23))
        mag_var.derive(lat, None, lon, None, alt_aal, start_datetime)
        ma_test.assert_masked_array_almost_equal(
            mag_var.array[0:10],
            np.ma.array([-6.064445460989708, -6.065693019716132, -6.066940578442557,
                         -6.068188137168981, -6.069435695895405, -6.070683254621829,
                         -6.071930813348254, -6.073178372074678, -6.074425930801103,
                         -6.075673489527527]))
        # Test with Coarse parameters.
        mag_var.derive(None, lat, None, lon, alt_aal, start_datetime)
        ma_test.assert_masked_array_almost_equal(
            mag_var.array[300:310],
            np.ma.array([-6.506129083442324, -6.507848687633959, -6.509568291825593,
                         -6.511287896017228, -6.513007500208863, -6.514727104400498,
                         -6.516446708592133, -6.518166312783767, -6.519885916975402,
                         -6.521605521167037]))

class TestMagneticVariationFromRunway(unittest.TestCase):
    def test_can_operate(self):
        opts = MagneticVariationFromRunway.get_operational_combinations()
        self.assertEqual(opts,
                    [('HDF Duration',
                     'Heading During Takeoff',
                     'Heading During Landing',
                     'FDR Takeoff Runway',
                     'FDR Landing Runway',
                     )])
        
    def test_derive_both_runways(self):
        toff_rwy = {'end': {'elevation': 10,
                            'latitude': 52.7100630002283,
                            'longitude': -8.907803520515461},
                    'start': {'elevation': 43,
                              'latitude': 52.69327604095164,
                              'longitude': -8.943465355819775},
                    'strip': {'id': 2014, 'length': 10495, 
                              'surface': 'ASP', 'width': 147}}
        land_rwy = {'end': {'elevation': 374,
                            'latitude': 49.024719,
                            'longitude': 2.524892},
                    'start': {'elevation': 377,
                              'latitude': 49.026694,
                              'longitude': 2.561689},
                    'strip': {'id': 2322, 'length': 8858,
                              'surface': 'ASP', 'width': 197}}
        mag_var_rwy = MagneticVariationFromRunway()
        mag_var_rwy.derive(
            A('HDF Duration', 14272),
            KPV([KeyPointValue(index=62.143, value=58.014, name='Heading During Takeoff')]),
            KPV([KeyPointValue(index=213.869, value=266.5128, name='Heading During Landing')]),
            A('FDR Takeoff Runway', toff_rwy),
            A('FDR Landing Runway', land_rwy)
        )
        # 0 to takeoff index variation
        self.assertAlmostEqual(mag_var_rwy.array[0], -5.87, places=2)
        self.assertAlmostEqual(mag_var_rwy.array[62], -5.87, places=2)
        # landing index to end
        self.assertAlmostEqual(mag_var_rwy.array[213], -1.18, places=2)
        self.assertAlmostEqual(mag_var_rwy.array[-1], -1.18, places=2)
        
    def test_derive_only_takeoff_available(self):
        toff_rwy = {'end': {'elevation': 10,
                            'latitude': 52.7100630002283,
                            'longitude': -8.907803520515461},
                    'start': {'elevation': 43,
                              'latitude': 52.69327604095164,
                              'longitude': -8.943465355819775},
                    'strip': {'id': 2014, 'length': 10495, 
                              'surface': 'ASP', 'width': 147}}
        land_rwy = {# MISSING VITAL LAT/LONG INFORMATION
                    'strip': {'id': 2322, 'length': 8858,
                              'surface': 'ASP', 'width': 197}}
        mag_var_rwy = MagneticVariationFromRunway()
        mag_var_rwy.derive(
            A('HDF Duration', 14272),
            KPV([KeyPointValue(index=62.143, value=58.014, name='Heading During Takeoff')]),
            KPV([KeyPointValue(index=213.869, value=266.5128, name='Heading During Landing')]),
            A('FDR Takeoff Runway', toff_rwy),
            A('FDR Landing Runway', land_rwy)
        )
        # 0 to takeoff index variation
        self.assertAlmostEqual(mag_var_rwy.array[0], -5.87, places=2)
        self.assertAlmostEqual(mag_var_rwy.array[62], -5.87, places=2)
        # landing index to end
        self.assertAlmostEqual(mag_var_rwy.array[213], -5.87, places=2)
        self.assertAlmostEqual(mag_var_rwy.array[-1], -5.87, places=2)
    
    def test_derive_landing_rollover(self):
        toff_rwy = {'end': {'elevation': 354, 'latitude': 32.697986, 'longitude': -83.644942},
                    'id': 13451,
                    'identifier': '05',
                    'magnetic_heading': 52.9,
                    'start': {'elevation': 354, 'latitude': 32.686022, 'longitude': -83.660633},
                    'strip': {'id': 6726, 'length': 6501, 'surface': 'ASP', 'width': 150}}
        land_rwy = {'end': {'elevation': 1289,
                            'latitude': 36.9130564484,
                            'longitude': -94.012504957},
                    'id': 19503,
                    'identifier': '36',
                    'magnetic_heading': 359.0,
                    'start': {'elevation': 1289,
                              'latitude': 36.899355805,
                              'longitude': -94.0129958013},
                    'strip': {'id': 12132, 'length': 5000, 'surface': 'ASPH', 'width': 75}}
        mag_var_rwy = MagneticVariationFromRunway()
        mag_var_rwy.derive(
            A('HDF Duration', 9336),
            KPV([KeyPointValue(index=148.9833984375, value=53.00950000000029, name='Heading During Takeoff')]),
            KPV([KeyPointValue(index=1580.7333984375, value=359.8320000000002, name='Heading During Landing')]),
            A('FDR Takeoff Runway', toff_rwy),
            A('FDR Landing Runway', land_rwy)
        )
        # 0 to takeoff index variation
        self.assertTrue(np.all(mag_var_rwy.array[:149] == -5.1902489670943481))
        # landing index to end
        self.assertTrue(np.all(mag_var_rwy.array[1580:] == 1.8087827938368832))


class TestPitchRate(unittest.TestCase):
    @unittest.skip('Test Not Implemented')
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')
        
    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestRelief(unittest.TestCase):
    @unittest.skip('Test Not Implemented')
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')
        
    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestRoll(unittest.TestCase):
    def test_can_operate(self):
        opts = Roll.get_operational_combinations()
        self.assertTrue(('Heading Continuous', 'Altitude AAL',) in opts)
  
    def test_derive(self):
        time = np.arange(100)
        two_time = np.arange(200)
        zero = np.array([0]*100)
        ht_values = np.concatenate([zero, 2000.0*(1.0-np.cos(two_time*np.pi*0.01)), zero])
        ht=P('Altitude AAL', array=np.ma.array(ht_values), frequency=2.0)
        hdg_values = np.concatenate([20.0*(np.sin(time*np.pi*0.03)), zero])
        hdg_values += 120 # Datum heading offset
        hdg=P('Heading', array=np.ma.array(hdg_values), frequency=1.0)
        herc = A('Frame', 'L382-Hercules')
        derroll=Roll()
        derroll.derive(None, None, hdg, ht, herc)
        self.assertLess(derroll.array[40], 0.25)
        self.assertLess(np.ma.max(derroll.array),13.0)
        self.assertGreater(np.ma.max(derroll.array),11.0)

class TestRollRate(unittest.TestCase):
    def test_can_operate(self):
        opts = RollRate.get_operational_combinations()
        self.assertTrue(('Roll',) in opts)
        
    def test_derive(self):
        roll = P(array=[0,2,4,6,8,10,12], name='Roll', frequency=2.0)
        rr = RollRate()
        rr.derive(roll)
        expected=np_ma_ones_like(roll.array)*4.0
        ma_test.assert_array_equal(expected[2:4], rr.array[2:4]) # Differential process blurs ends of the array, so just test the core part.


class TestRudderPedal(unittest.TestCase):
    def test_can_operate(self):
        opts = RudderPedal.get_operational_combinations()
        self.assertTrue(('Rudder Pedal (L)',) in opts)
        self.assertTrue(('Rudder Pedal (R)',) in opts)
        self.assertTrue(('Rudder Pedal (L)', 'Rudder Pedal (R)') in opts)
        self.assertTrue(('Rudder Pedal Potentiometer',) in opts)
        self.assertTrue(('Rudder Pedal Synchro',) in opts)
        self.assertTrue(('Rudder Pedal Potentiometer', 'Rudder Pedal Synchro') in opts)
        
        
    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestSlatAngle(unittest.TestCase):
    def test_can_operate(self):
        self.assertEqual(
            SlatAngle.get_operational_combinations(),
            [('Slat Angle (L)',), ('Slat Angle (R)',), ('Slat Angle (L)', 'Slat Angle (R)')])
    
    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestSlopeToLanding(unittest.TestCase):
    @unittest.skip('Test Not Implemented')
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')
        
    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestSpeedbrake(unittest.TestCase):
    def test_can_operate(self):
        family = A(name='Family', value='B737 Classic')
        self.assertTrue(Speedbrake.can_operate(('Spoiler (2)', 'Spoiler (7)'),
                                               family=family))
        family = A(name='Family', value='B737 NG')
        self.assertTrue(Speedbrake.can_operate(('Spoiler (4)', 'Spoiler (9)'),
                                               family=family))
        family = A(name='Family', value='A320')
        self.assertTrue(Speedbrake.can_operate(('Spoiler (2)', 'Spoiler (7)'),
                                               family=family))
        family = A(name='Family', value='B787')
        self.assertTrue(Speedbrake.can_operate(('Spoiler (1)', 'Spoiler (14)'),
                                               family=family))
        family = A(name='Family', value='Learjet')
        self.assertTrue(Speedbrake.can_operate(('Spoiler (L)', 'Spoiler (R)'),
                                               family=family))
        family = A(name='Family', value='CRJ 900')
        self.assertTrue(Speedbrake.can_operate(
            ('Spoiler (L) Inboard', 'Spoiler (L) Outboard',
             'Spoiler (R) Inboard', 'Spoiler (R) Outboard'), family=family))
        family = A(name='Family', value='Phenom 300')
        self.assertTrue(Speedbrake.can_operate(('Spoiler (L)', 'Spoiler (R)'),
                                               family=family))
        family = A(name='Family', value='ERJ-190/195')
        self.assertTrue(Speedbrake.can_operate(
            ('Spoiler (L) Inboard', 'Spoiler (L) Middle', 'Spoiler (L) Outboard',
             'Spoiler (R) Inboard', 'Spoiler (R) Middle', 'Spoiler (R) Outboard'), family=family))

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestSAT(unittest.TestCase):
    # Note: the core function machtat2sat is tested by the library test.
    @unittest.skip('Test Not Implemented')
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')
        
    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestTAT(unittest.TestCase):
    @unittest.skip('Test Not Implemented')
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')
        
    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestTailwind(unittest.TestCase):
    @unittest.skip('Test Not Implemented')
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')
        
    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestThrustAsymmetry(unittest.TestCase):
    def test_can_operate(self):
        opts = ThrustAsymmetry.get_operational_combinations()
        expected = [
            ('Eng (*) EPR Max', 'Eng (*) EPR Min'),
            ('Eng (*) N1 Max', 'Eng (*) N1 Min'),
            ]
        for exp in expected:
            self.assertIn(exp, opts, msg=exp)
        
    def test_derive_n1(self):
        n1_max = P('Eng (*) N1 Max', np.arange(10, 30))
        n1_min = P('Eng (*) N1 Min', np.arange(8, 28))
        asym = ThrustAsymmetry()
        asym.derive(None, None, n1_max=n1_max, n1_min=n1_min)
        self.assertEqual(len(asym.array), len(n1_max.array))
        uniq = unique_values(asym.array.astype(int))
        # there should be all 20 values being 2 out
        self.assertEqual(uniq, {2: 16})  # 3 values masked out
        
    def test_derive_epr_and_n1(self):
        n1_max = P('Eng (*) N1 Max', np.arange(10, 30))
        n1_min = P('Eng (*) N1 Min', np.arange(8, 28))
        # create decimal array in the range of 0 to 2
        epr_max = P('Eng (*) EPR Max', np.arange(0.0, 1.8, step=0.2))
        epr_min = P('Eng (*) EPR Min', np.arange(0.2, 2.0, step=0.2))
        asym = ThrustAsymmetry()
        asym.get_derived((epr_max, epr_min, n1_max, n1_min))
        self.assertEqual(len(asym.array), len(epr_max.array))
        self.assertEqual(asym.array[2], -20)
        self.assertAlmostEqual(asym.array[-3], -20)
        


class TestThrottleLevers(unittest.TestCase):
    def test_can_operate(self):
        self.assertEqual(ThrottleLevers.get_operational_combinations(),
                         [('Eng (1) Throttle Lever',),
                          ('Eng (2) Throttle Lever',),
                          ('Eng (1) Throttle Lever', 'Eng (2) Throttle Lever')])
        
    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestTurbulence(unittest.TestCase):
    @unittest.skip('Test Not Implemented')
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')

    def test_derive(self):
        accel = np.ma.array([1]*40+[2]+[1]*40)
        turb = TurbulenceRMSG()
        turb.derive(P('Acceleration Vertical', accel, frequency=8))
        expected = np.array([0]*20+[0.156173762]*41+[0]*20)
        np.testing.assert_array_almost_equal(expected, turb.array.data)


class TestVOR1Frequency(unittest.TestCase):
    @unittest.skip('Test Not Implemented')
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')
        
    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestVOR2Frequency(unittest.TestCase):
    @unittest.skip('Test Not Implemented')
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')
        
    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestVerticalSpeedInertial(unittest.TestCase):
    @unittest.skip('Test Not Implemented')
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')
        
    def test_derive(self):
        time = np.arange(100)
        zero = np.array([0]*50)
        acc_values = np.concatenate([zero, np.cos(time*np.pi*0.02), zero])
        vel_values = np.concatenate([zero, np.sin(time*np.pi*0.02), zero])
        ht_values = np.concatenate([zero, 1.0-np.cos(time*np.pi*0.02), zero])
        
        # For a 0-400ft leap over 100 seconds, the scaling is 200ft amplitude and 2*pi/100 for each differentiation.
        amplitude = 200.0
        diff = 2.0 * np.pi / 100.0
        ht_values *= amplitude
        vel_values *= amplitude * diff * 60.0 # fpm
        acc_values *= amplitude * diff**2.0 / GRAVITY_IMPERIAL # g
        
        #import wx
        #import matplotlib.pyplot as plt
        #plt.plot(acc_values,'k')
        #plt.plot(vel_values,'b')
        #plt.plot(ht_values,'r')
        #plt.show()
        
        az = P('Acceleration Vertical', acc_values)
        alt_std = P('Altitude STD Smoothed', ht_values + 30.0) # Pressure offset
        alt_rad = P('Altitude STD Smoothed', ht_values-2.0) #Oleo compression
        fast = buildsection('Fast', 10, len(acc_values)-10)

        vsi = VerticalSpeedInertial()
        vsi.derive(az, alt_std, alt_rad, fast)
        
        expected = vel_values

        # Just check the graphs are similar in shape - there will always be
        # errors because of the integration technique used.
        np.testing.assert_almost_equal(vsi.array, expected, decimal=-2)


class TestRudder(unittest.TestCase):
    def test_can_operate(self):
        opts = Rudder.get_operational_combinations()
        # Set up currently for 777 which has to have three parts.
        #self.assertIn(('Rudder (Upper)'), opts)
        #self.assertIn(('Rudder (Middle)'), opts)
        #self.assertIn(('Rudder (Lower)'), opts)
        self.assertIn(('Rudder (Upper)', 'Rudder (Middle)', 'Rudder (Lower)'), opts)

    def test_derive(self):
        upper = P('Rudder (Upper)', array=np.ma.array([0]*7))
        upper.array[1] = 3
        middle = P('Rudder (Middle)', array=np.ma.array([0]*7))
        middle.array[3] = 6
        lower = P('Rudder (Lower)', array=np.ma.array([0]*7))
        lower.array[5] = 9
        rud = Rudder()
        rud.derive(upper, middle, lower)
        np.testing.assert_almost_equal(rud.array.data, [0,1,0,2,0,3,0])
        
class TestStabilizer(unittest.TestCase):
    def test_can_operate(self):
        opts = Stabilizer.get_operational_combinations()
        # Set up currently for 777 which has to have four parts.
        self.assertIn(('Stabilizer (1)', 'Stabilizer (2)', 'Stabilizer (3)', 'Frame'), opts)

    def test_basic(self):
        s1 = P('Stabilizer (1)', array=np.ma.array([0]*7))
        s1.array[1] = 3
        s2 = P('Stabilizer (2)', array=np.ma.array([0]*7))
        s2.array[3] = 6
        s3 = P('Stabilizer (3)', array=np.ma.array([0]*7))
        s3.array[5] = 9
        frame = A('Frame', '777')
        stab = Stabilizer()
        stab.derive(s1, s2, s3, frame)
        result = (np.array([0,1,0,2,0,3,0]) * 0.0503) - 3.4629
        # Blend will null the end values, so only test excluding these.
        np.testing.assert_almost_equal(stab.array.data[1:-1], result[1:-1])
        
    def test_wrong_frame(self):
        s1 = P('Stabilizer (1)', array=np.ma.array([0]*7))
        frame = A('Frame', '787')
        stab = Stabilizer()
        self.assertRaises(ValueError, stab.derive, s1, s1, s1, frame)


class TestWheelSpeed(unittest.TestCase):
    def test_can_operate(self):
        opts = WheelSpeed.get_operational_combinations()
        self.assertEqual(opts, 
                         [('Wheel Speed (L)', 'Wheel Speed (R)'),
                          #('Wheel Speed (L)', 'Wheel Speed (C)', 'Wheel Speed (R)'),
                          ])
         
    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        pass


class TestWheelSpeedLeft(unittest.TestCase):
    def test_can_operate(self):
        opts = WheelSpeedLeft.get_operational_combinations()
        self.assertIn(('Wheel Speed (L) (1)', 'Wheel Speed (L) (2)'), opts)
        self.assertIn(('Wheel Speed (L) (1)', 'Wheel Speed (L) (2)', 'Wheel Speed (L) (3)', 'Wheel Speed (L) (4)'), opts)

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        pass


class TestWheelSpeedRight(unittest.TestCase):
    def test_can_operate(self):
        opts = WheelSpeedRight.get_operational_combinations()
        self.assertIn(('Wheel Speed (R) (1)', 'Wheel Speed (R) (2)'), opts)
        self.assertIn(('Wheel Speed (R) (1)', 'Wheel Speed (R) (2)', 'Wheel Speed (R) (3)', 'Wheel Speed (R) (4)'), opts)
         
    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        pass


class TestWindDirectionContinuous(unittest.TestCase):
    @unittest.skip('Test Not Implemented')
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')
        
    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestWindDirection(unittest.TestCase):
    def test_can_operate(self):
        self.assertTrue(WindDirection.can_operate(('Wind Direction (1)',)))
        self.assertTrue(WindDirection.can_operate(('Wind Direction (2)',)))
        self.assertTrue(WindDirection.can_operate(('Wind Direction (1)',
                                                   'Wind Direction (2)',)))
        self.assertTrue(WindDirection.can_operate(('Wind Direction True',
                                                   'Magnetic Variation',)))
        self.assertTrue(WindDirection.can_operate((
            'Wind Direction (1)', 'Wind Direction (2)', 'Wind Direction True',
            'Magnetic Variation')))
    
    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestWindDirectionTrue(unittest.TestCase):
    def test_can_operate(self):
        self.assertEqual(WindDirectionTrue.get_operational_combinations(),
                         [('Wind Direction', 'Magnetic Variation From Runway'), 
                          ('Wind Direction', 'Magnetic Variation'),
                          ('Wind Direction', 'Magnetic Variation From Runway', 'Magnetic Variation')])
    
    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestCoordinatesSmoothed(TemporaryFileTest, unittest.TestCase):
    def setUp(self):
        self.approaches = App('Approach Information',
            items=[ApproachItem('GO_AROUND', slice(3198.0, 3422.0),
                            ils_freq=108.55,
                            gs_est=slice(3200, 3390),
                            loc_est=slice(3199, 3445),
                            airport={'code': {'iata': 'KDH', 'icao': 'OAKN'},
                                     'distance': 2.483270162497824,
                                     'elevation': 3301,
                                     'id': 3279,
                                     'latitude': 31.5058,
                                     'location': {'country': 'Afghanistan'},
                                     'longitude': 65.8478,
                                     'magnetic_variation': 'E001590 0506',
                                     'name': 'Kandahar'},
                            runway={'end': {'elevation': 3294,
                                            'latitude': 31.497511,
                                            'longitude': 65.833933},
                                    'id': 44,
                                    'identifier': '23',
                                    'magnetic_heading': 232.9,
                                    'start': {'elevation': 3320,
                                              'latitude': 31.513997,
                                              'longitude': 65.861714},
                                    'strip': {'id': 22,
                                              'length': 10532,
                                              'surface': 'ASP',
                                              'width': 147}}),
                   ApproachItem('LANDING', slice(12928.0, 13440.0),
                            ils_freq=111.3,
                            gs_est=slice(13034, 13262),
                            loc_est=slice(12929, 13347),
                            turnoff=13362.455208333333,
                            airport={'code': {'iata': 'DXB', 'icao': 'OMDB'},
                                     'distance': 1.6842014290716794,
                                     'id': 3302,
                                     'latitude': 25.2528,
                                     'location': {'city': 'Dubai',
                                                  'country': 'United Arab Emirates'},
                                     'longitude': 55.3644,
                                     'magnetic_variation': 'E001315 0706',
                                     'name': 'Dubai Intl'},
                            runway={'end': {'latitude': 25.262131, 'longitude': 55.347572},
                                    'glideslope': {'angle': 3.0,
                                                   'latitude': 25.246333,
                                                   'longitude': 55.378417,
                                                   'threshold_distance': 1508},
                                    'id': 22,
                                    'identifier': '30L',
                                    'localizer': {'beam_width': 4.5,
                                                  'frequency': 111300.0,
                                                  'heading': 300,
                                                  'latitude': 25.263139,
                                                  'longitude': 55.345722},
                                    'magnetic_heading': 299.7,
                                    'start': {'latitude': 25.243322, 'longitude': 55.381519},
                                    'strip': {'id': 11,
                                              'length': 13124,
                                              'surface': 'ASP',
                                              'width': 150}})])
        
        self.toff = [Section(name='Takeoff', 
                             slice=slice(372, 414, None), 
                             start_edge=371.32242063492066, 
                             stop_edge=413.12204760355382)]
        
        self.toff_rwy = A(name = 'FDR Takeoff Runway',
                          value = {'end': {'elevation': 4843, 
                                           'latitude': 34.957972, 
                                           'longitude': 69.272944},
                                   'id': 41,
                                   'identifier': '03',
                                   'magnetic_heading': 26.0,
                                   'start': {'elevation': 4862, 
                                             'latitude': 34.934306, 
                                             'longitude': 69.257},
                                   'strip': {'id': 21, 
                                             'length': 9852, 
                                             'surface': 'CON', 
                                             'width': 179}})

        self.source_file_path = os.path.join(
            test_data_path, 'flight_with_go_around_and_landing.hdf5')
        super(TestCoordinatesSmoothed, self).setUp()

    # Skipped by DJ's advice: too many changes withoud updating the test
    @unittest.skip('Test Out Of Date')
    def test__adjust_track_precise(self):
        with hdf_file(self.test_file_path) as hdf:
            lon = hdf['Longitude']
            lat = hdf['Latitude']
            ils_loc =hdf['ILS Localizer']
            app_range = hdf['ILS Localizer Range']
            gspd = hdf['Groundspeed']
            hdg = hdf['Heading True Continuous']
            tas = hdf['Airspeed True']
            rot = hdf['Rate Of Turn']

        precision = A(name='Precise Positioning', value = True)
        mobile = Mobile()
        mobile.get_derived((rot, gspd))
        
        cs = CoordinatesSmoothed()    
        lat_new, lon_new = cs._adjust_track(
            lon, lat, ils_loc, app_range, hdg, gspd, tas, 
            self.toff, self.toff_rwy, self.approaches, mobile, precision)
        
        chunks = np.ma.clump_unmasked(lat_new)
        self.assertEqual(len(chunks),3)
        self.assertEqual(chunks,[slice(44, 372, None), 
                                 slice(3200, 3445, None), 
                                 slice(12930, 13424, None)])
        
    # Skipped by DJ's advice: too many changes withoud updating the test
    @unittest.skip('Test Out Of Date')
    def test__adjust_track_imprecise(self):
        with hdf_file(self.test_file_path) as hdf:
            lon = hdf['Longitude']
            lat = hdf['Latitude']
            ils_loc =hdf['ILS Localizer']
            app_range = hdf['ILS Localizer Range']
            gspd = hdf['Groundspeed']
            hdg = hdf['Heading True Continuous']
            tas = hdf['Airspeed True']
            rot = hdf['Rate Of Turn']

        precision = A(name='Precise Positioning', value = False)
        
        mobile = Mobile()
        mobile.get_derived((rot, gspd))
        cs = CoordinatesSmoothed()    
        lat_new, lon_new = cs._adjust_track(
            lon, lat, ils_loc, app_range, hdg, gspd, tas, 
            self.toff, self.toff_rwy, self.approaches, mobile, precision)
        
        chunks = np.ma.clump_unmasked(lat_new)
        self.assertEqual(len(chunks),2)
        self.assertEqual(chunks,[slice(44,414),slice(12930,13424)])
        

        #import matplotlib.pyplot as plt
        #plt.plot(lat_new, lon_new)
        #plt.show()
        #plt.plot(lon.array, lat.array)
        #plt.show()

    # Skipped by DJ's advice: too many changes withoud updating the test
    @unittest.skip('Test Out Of Date')
    def test__adjust_track_visual(self):
        with hdf_file(self.test_file_path) as hdf:
            lon = hdf['Longitude']
            lat = hdf['Latitude']
            ils_loc =hdf['ILS Localizer']
            app_range = hdf['ILS Localizer Range']
            gspd = hdf['Groundspeed']
            hdg = hdf['Heading True Continuous']
            tas = hdf['Airspeed True']
            rot = hdf['Rate Of Turn']

        precision = A(name='Precise Positioning', value = False)
        mobile = Mobile()
        mobile.get_derived((rot, gspd))
        
        self.approaches.value[0].pop('ILS localizer established')
        self.approaches.value[1].pop('ILS localizer established')
        # Don't need to pop the glideslopes as these won't be looked for.
        cs = CoordinatesSmoothed()
        lat_new, lon_new = cs._adjust_track(
            lon, lat, ils_loc, app_range, hdg, gspd, tas, 
            self.toff, self.toff_rwy, self.approaches, mobile, precision)
        
        chunks = np.ma.clump_unmasked(lat_new)
        self.assertEqual(len(chunks),2)
        self.assertEqual(chunks,[slice(44,414),slice(12930,13424)])


class TestApproachRange(TemporaryFileTest, unittest.TestCase):
    def setUp(self):
        self.approaches = App(items=[
            ApproachItem('GO_AROUND', slice(3198, 3422),
                     ils_freq=108.55,
                     gs_est=slice(3200, 3390),
                     loc_est=slice(3199, 3445),
                     airport={'code': {'iata': 'KDH', 'icao': 'OAKN'},
                              'distance': 2.483270162497824,
                              'elevation': 3301,
                              'id': 3279,
                              'latitude': 31.5058,
                              'location': {'country': 'Afghanistan'},
                              'longitude': 65.8478,
                              'magnetic_variation': 'E001590 0506',
                              'name': 'Kandahar'},
                     runway={'end': {'elevation': 3294,
                                     'latitude': 31.497511,
                                     'longitude': 65.833933},
                             'id': 44,
                             'identifier': '23',
                             'magnetic_heading': 232.9,
                             'start': {'elevation': 3320,
                                       'latitude': 31.513997,
                                       'longitude': 65.861714},
                             'strip': {'id': 22,
                                       'length': 10532,
                                       'surface': 'ASP',
                                       'width': 147}}),
            ApproachItem('LANDING', slice(12928, 13440),
                     ils_freq=111.3,
                     gs_est=slice(13034, 13262),
                     loc_est=slice(12929, 13347),
                     turnoff=13362.455208333333,
                     airport={'code': {'iata': 'DXB', 'icao': 'OMDB'},
                              'distance': 1.6842014290716794,
                              'id': 3302,
                              'latitude': 25.2528,
                              'location': {'city': 'Dubai',
                                           'country': 'United Arab Emirates'},
                              'longitude': 55.3644,
                              'magnetic_variation': 'E001315 0706',
                              'name': 'Dubai Intl'},
                     runway={'end': {'latitude': 25.262131, 'longitude': 55.347572},
                             'glideslope': {'angle': 3.0,
                                            'latitude': 25.246333,
                                            'longitude': 55.378417,
                                            'threshold_distance': 1508},
                             'id': 22,
                             'identifier': '30L',
                             'localizer': {'beam_width': 4.5,
                                           'frequency': 111300.0,
                                           'heading': 300,
                                           'latitude': 25.263139,
                                           'longitude': 55.345722},
                             'magnetic_heading': 299.7,
                             'start': {'latitude': 25.243322, 'longitude': 55.381519},
                             'strip': {'id': 11,
                                       'length': 13124,
                                       'surface': 'ASP',
                                       'width': 150}})])
        
        self.toff = Section(name='Takeoff', 
                       slice=slice(372, 414, None), 
                       start_edge=371.32242063492066, 
                       stop_edge=413.12204760355382)
        
        self.toff_rwy = A(name='FDR Takeoff Runway',
                          value={'end': {'elevation': 4843, 
                                         'latitude': 34.957972, 
                                         'longitude': 69.272944},
                                 'id': 41,
                                 'identifier': '03',
                                 'magnetic_heading': 26.0,
                                 'start': {'elevation': 4862,
                                           'latitude': 34.934306,
                                           'longitude': 69.257},
                                 'strip': {'id': 21,
                                           'length': 9852,
                                           'surface': 'CON',
                                           'width': 179}})

        self.source_file_path = os.path.join(
            test_data_path, 'flight_with_go_around_and_landing.hdf5')
        super(TestApproachRange, self).setUp()

    def test_can_operate(self):
        operational_combinations = ApproachRange.get_operational_combinations()
        self.assertTrue(('Heading True Continuous', 'Airspeed True', 'Altitude AAL', 'Approach Information') in operational_combinations, msg="Missing 'Heading True' combination")
        self.assertTrue(('Track True Continuous', 'Airspeed True', 'Altitude AAL', 'Approach Information') in operational_combinations, msg="Missing 'Track True' combination")
        self.assertTrue(('Track Continuous', 'Airspeed True', 'Altitude AAL', 'Approach Information') in operational_combinations, msg="Missing 'Track' combination")
        self.assertTrue(('Heading Continuous', 'Airspeed True', 'Altitude AAL', 'Approach Information') in operational_combinations, msg="Missing 'Heading' combination")

    def test_range_basic(self):
        with hdf_file(self.test_file_path) as hdf:
            hdg = hdf['Heading True']
            tas = hdf['Airspeed True']
            alt = hdf['Altitude AAL']
            glide = hdf['ILS Glideslope']
        
        ar = ApproachRange()    
        ar.derive(None, glide, None, None, None, hdg, tas, alt, self.approaches)
        result = ar.array
        chunks = np.ma.clump_unmasked(result)
        self.assertEqual(len(chunks),2)
        self.assertEqual(chunks,[slice(3198, 3422, None), 
                                 slice(12928, 13440, None)])
        
    def test_range_full_param_set(self):
        with hdf_file(self.test_file_path) as hdf:
            hdg = hdf['Track True']
            tas = hdf['Airspeed True']
            alt = hdf['Altitude AAL']
            glide = hdf['ILS Glideslope']
            gspd = hdf['Groundspeed']
        
        ar = ApproachRange()    
        ar.derive(gspd, glide, None, None, hdg, None, tas, alt, self.approaches)
        result = ar.array
        chunks = np.ma.clump_unmasked(result)
        self.assertEqual(len(chunks),2)
        self.assertEqual(chunks,[slice(3198, 3422, None), 
                                 slice(12928, 13440, None)])


class TestZeroFuelWeight(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = ZeroFuelWeight
        self.operational_combinations = [
            ('HDF Duration', 'Fuel Qty', 'Gross Weight'),
            ('HDF Duration', 'Dry Operating Weight',),
            ('HDF Duration', 'Dry Operating Weight', 'Payload'),
        ]
        self.duration = A('HDF Duration', 10)

    def test_derive_fuel_qty_gross_wgt(self):
        fuel_qty = P('Fuel Qty', np.ma.array([1, 2, 3, 4]))
        gross_wgt = P('Gross Weight', np.ma.array([11, 12, 13, 14]))
        zfw = ZeroFuelWeight()
        zfw.derive(fuel_qty, gross_wgt, None, None, self.duration)
        self.assertTrue((zfw.array == 10).all())
    
    def test_derive_dry_operating_wgt(self):
        dry_operating_wgt = A('Dry Operating Weight', 100000)
        zfw = ZeroFuelWeight()
        zfw.derive(None, None, dry_operating_wgt, None, self.duration)
        self.assertTrue((zfw.array == dry_operating_wgt.value).all())
    
    def test_derive_dry_operating_wgt_payload(self):
        dry_operating_wgt = A('Dry Operating Weight', 100000)
        payload = A('Payload', None)
        zfw = ZeroFuelWeight()
        zfw.derive(None, None, dry_operating_wgt, payload, self.duration)
        self.assertTrue((zfw.array == dry_operating_wgt.value).all())
        
        payload = A('Payload', 1000)
        zfw = ZeroFuelWeight()
        zfw.derive(None, None, dry_operating_wgt, payload, self.duration)
        self.assertTrue((zfw.array == 101000).all())


class TestGrossWeight(unittest.TestCase):
    def test_can_operate(self):
        combinations = GrossWeight.get_operational_combinations()
        self.assertTrue(('Zero Fuel Weight', 'Fuel Qty') in combinations)
        self.assertTrue(('HDF Duration', 'AFR Landing Gross Weight',) in combinations)
        self.assertTrue(('HDF Duration',
                         'AFR Landing Gross Weight',
                         'AFR Takeoff Gross Weight',
                         'Touchdown',
                         'Liftoff') in combinations)
    
    def test_derive_fuel_qty_zfw(self):
        fq = P('Fuel Qty', array=np.ma.array([40,30,20,10]))
        zfw = P('Zero Fuel Weight', array=np.ma.array([990, 1000, 1000, 1100]))
        node = GrossWeight()
        node.derive(zfw, fq, None, None, None, None, None)
        self.assertEqual(node.array.tolist(), [1040, 1030, 1020, 1010])
    
    def test_derive_afr_land_wgt(self):
        duration = A('HDF Duration', 100)
        afr_land_wgt = A('AFR Landing Gross Weight', 1000)
        
        node = GrossWeight()
        node.derive(None, None, duration, afr_land_wgt, None, None, None)
        
        self.assertEqual(node.array.tolist(), [1000] * 100)
    
    def test_derive_afr_interpolated_wgt(self):
        duration = A('HDF Duration', 100)
        afr_takeoff_wgt = A('AFR Takeoff Gross Weight', 2000)
        afr_land_wgt = A('AFR Landing Gross Weight', 1000)
        liftoffs = KTI('Liftoff', items=[KeyTimeInstance(10)])
        touchdowns = KTI('Touchdown', items=[KeyTimeInstance(90)])
        
        node = GrossWeight()
        node.derive(None, None, duration, afr_land_wgt, afr_takeoff_wgt,
                    touchdowns, liftoffs)
        
        result = node.array.tolist()
        self.assertEqual(result[:10], [None] * 10)
        self.assertEqual(result[-10:], [None] * 10)
        self.assertEqual(result[10], 2000)
        self.assertEqual(result[-11], 1000)
        self.assertEqual(result[:10], [None] * 10)
        self.assertAlmostEqual(result[50], 1493.7, 1)
    
    def test_using_zero_fuel_weight(self):
        fuel_qty_array = np.ma.arange(10000, 4000, -100)
        fuel_qty_array[10] *= 0.9
        fuel_qty = P('Fuel Qty', array=fuel_qty_array)
        
        zfw_array = np_ma_ones_like(fuel_qty_array) * 17400
        zfw = P('Zero Fuel Weight', array=zfw_array)
        
        gw = GrossWeight()
        gw.derive(zfw, fuel_qty, None, None, None, None, None)
        
        self.assertEquals(len(gw.array), 60)
        self.assertEqual(gw.array[4], (10000-(4*100)+17400))
        
        # check fuel quantity decrease at liftoff by 90% 
        # only has limited 4% effect on array
        self.assertLess(gw.array[10], (10000-(10*100)+17400))
        self.assertGreater(gw.array[10], (10000-(10*100)+17400)*0.96) 
        self.assertEqual(gw.array[25], (10000-(25*100)+17400))


##############################################################################
# Velocity Speeds


########################################
# Takeoff Safety Speed (V2)


class TestV2(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = V2
        self.airspeed = P('Airspeed', np.ma.repeat(200, 1280))
        self.liftoffs = KTI(name='Liftoff', items=[
            KeyTimeInstance(name='Liftoff', index=20),
            KeyTimeInstance(name='Liftoff', index=860),
        ])
        self.climbs = KTI(name='Climb Start', items=[
            KeyTimeInstance(name='Climb Start', index=420),
            KeyTimeInstance(name='Climb Start', index=1060),
        ])

    def test_can_operate(self):
        # AFR:
        self.assertTrue(self.node_class.can_operate(
            ('Airspeed', 'AFR V2', 'Liftoff', 'Climb Start'),
            afr_v2=A('AFR V2', 120),
        ))
        self.assertFalse(self.node_class.can_operate(
            ('Airspeed', 'AFR V2', 'Liftoff', 'Climb Start'),
            afr_v2=A('AFR V2', 70),
        ))
        # Embraer:
        self.assertTrue(self.node_class.can_operate(
            ('Airspeed', 'V2-Vac', 'Liftoff', 'Climb Start'),
        ))
        # Airbus:
        self.assertTrue(self.node_class.can_operate(
            ('Airspeed', 'Airspeed Selected', 'Speed Control', 'Liftoff', 'Climb Start', 'Manufacturer'),
            manufacturer=A('Manufacturer', 'Airbus'),
        ))

    def test_derive__nothing(self):
        node = self.node_class()
        node.derive(self.airspeed, None, None, None, None, self.liftoffs, self.climbs, None)
        expected = np.ma.repeat(0, 1280)
        expected[expected == 0] = np.ma.masked
        ma_test.assert_masked_array_equal(node.array, expected)

    def test_derive__afr_v2(self):
        afr_v2 = A('AFR V2', 120)
        node = self.node_class()
        node.derive(self.airspeed, None, None, None, afr_v2, self.liftoffs, self.climbs, None)
        expected = np.ma.repeat((120, 0, 120, 0), (420, 120, 520, 220))
        expected[expected == 0] = np.ma.masked
        ma_test.assert_masked_array_equal(node.array, expected)

    def test_derive__airbus(self):
        manufacturer = A(name='Manufacturer', value='Airbus')
        spd_ctl = M('Speed Control', np.ma.repeat((1, 0), (320, 960)), values_mapping={0: 'Manual', 1: 'Auto'})
        spd_sel = P('Airspeed Selected', np.ma.repeat((400, 120, 170), (10, 630, 640)))
        spd_sel.array[:10] = np.ma.masked
        node = self.node_class()
        node.derive(self.airspeed, None, spd_sel, spd_ctl, None, self.liftoffs, self.climbs, manufacturer)
        expected = np.ma.repeat((120, 0), (420, 860))
        expected[expected == 0] = np.ma.masked
        ma_test.assert_masked_array_equal(node.array, expected)

    def test_derive__embraer(self):
        manufacturer = A(name='Manufacturer', value='Embraer')
        v2_vac = P('V2-Vac', np.ma.repeat(150, 1280))
        node = self.node_class()
        node.derive(self.airspeed, v2_vac, None, None, None, self.liftoffs, self.climbs, manufacturer)
        expected = np.ma.repeat((150, 0, 150, 0), (420, 120, 520, 220))
        expected[expected == 0] = np.ma.masked
        ma_test.assert_masked_array_equal(node.array, expected)


class TestV2Lookup(unittest.TestCase, NodeTest):

    class VSX(VelocitySpeed):
        '''
        Table for aircraft with undefined V2.
        '''
        tables = {}

    class VSC0(VelocitySpeed):
        '''
        Table for aircraft with configuration and fallback tables.
        '''
        weight_unit = ut.TONNE
        tables = {'v2': {
            'weight': ( 35,  40,  45,  50,  55,  60,  65),
           'Lever 1': (113, 119, 126, 132, 139, 145, 152),
        }}
        fallback = {'v2': {'Lever 2': 140}}

    class VSC1(VelocitySpeed):
        '''
        Table for aircraft with configuration and fallback tables only.
        '''
        weight_unit = None
        fallback = {'v2': {'Lever 1': 135}}

    class VSF0(VelocitySpeed):
        '''
        Table for aircraft with flap and fallback tables.
        '''
        weight_unit = ut.TONNE
        tables = {'v2': {
            'weight': ( 35,  40,  45,  50,  55,  60,  65),
                '15': (113, 119, 126, 132, 139, 145, 152),
        }}
        fallback = {'v2': {'5': 140}}

    class VSF1(VelocitySpeed):
        '''
        Table for aircraft with flap and fallback tables only.
        '''
        weight_unit = None
        fallback = {'v2': {'17.5': 135}}

    def setUp(self):
        self.node_class = V2Lookup
        self.airspeed = P('Airspeed', np.ma.repeat(200, 1280))
        self.weight = KPV(name='Gross Weight At Liftoff', items=[
            KeyPointValue(name='Gross Weight At Liftoff', index=20, value=54192.06),
            KeyPointValue(name='Gross Weight At Liftoff', index=860, value=44192.06),
        ])
        self.liftoffs = KTI(name='Liftoff', items=[
            KeyTimeInstance(name='Liftoff', index=20),
            KeyTimeInstance(name='Liftoff', index=860),
        ])
        self.climbs = KTI(name='Climb Start', items=[
            KeyTimeInstance(name='Climb Start', index=420),
            KeyTimeInstance(name='Climb Start', index=1060),
        ])

    @patch('analysis_engine.library.at')
    def test_can_operate(self, at):
        nodes = ('Airspeed', 'Liftoff', 'Climb Start',
                 'Model', 'Series', 'Family', 'Engine Series', 'Engine Type')
        keys = ('model', 'series', 'family', 'engine_type', 'engine_series')
        airbus = dict(izip(keys, self.generate_attributes('airbus')))
        boeing = dict(izip(keys, self.generate_attributes('boeing')))
        beechcraft = dict(izip(keys, self.generate_attributes('beechcraft')))
        # Assume that lookup tables are found correctly...
        at.get_vspeed_map.return_value = self.VSF0
        # Flap Lever w/ Weight:
        available = nodes + ('Flap Lever', 'Gross Weight At Liftoff')
        self.assertTrue(self.node_class.can_operate(available, **boeing))
        # Flap Lever (Synthetic) w/ Weight:
        available = nodes + ('Flap Lever (Synthetic)', 'Gross Weight At Liftoff')
        self.assertTrue(self.node_class.can_operate(available, **airbus))
        # Flap Lever w/o Weight:
        available = nodes + ('Flap Lever',)
        self.assertTrue(self.node_class.can_operate(available, **beechcraft))
        # Flap Lever (Synthetic) w/o Weight:
        available = nodes + ('Flap Lever (Synthetic)',)
        self.assertTrue(self.node_class.can_operate(available, **airbus))
        # Assume that lookup tables are not found correctly...
        at.get_vspeed_map.side_effect = (KeyError, self.VSX)
        available = nodes + ('Flap Lever', 'Gross Weight At Liftoff')
        for i in range(2):
            self.assertFalse(self.node_class.can_operate(available, **boeing))

    @patch('analysis_engine.library.at')
    def test_derive__flap_with_weight__standard(self, at):
        mapping = {f: str(f) for f in (0, 1, 2, 5, 10, 15, 25, 30, 40)}
        flap_lever = M('Flap Lever', np.ma.repeat(15, 1280), values_mapping=mapping)

        attributes = self.generate_attributes('boeing')
        at.get_vspeed_map.return_value = self.VSF0

        node = self.node_class()
        node.derive(flap_lever, None, self.airspeed, self.weight,
                    self.liftoffs, self.climbs, *attributes)

        attributes = (a.value for a in attributes)
        at.get_vspeed_map.assert_called_once_with(*attributes)
        expected = np.ma.repeat((138, 0, 125, 0), (420, 120, 520, 220))
        expected[expected == 0] = np.ma.masked
        ma_test.assert_masked_array_equal(node.array, expected)

    @patch('analysis_engine.library.at')
    def test_derive__flap_with_weight__fallback(self, at):
        mapping = {f: str(f) for f in (0, 1, 2, 5, 10, 15, 25, 30, 40)}
        flap_lever = M('Flap Lever', np.ma.repeat(5, 1280), values_mapping=mapping)

        attributes = self.generate_attributes('boeing')
        at.get_vspeed_map.return_value = self.VSF0

        node = self.node_class()
        node.derive(flap_lever, None, self.airspeed, self.weight,
                    self.liftoffs, self.climbs, *attributes)

        attributes = (a.value for a in attributes)
        at.get_vspeed_map.assert_called_once_with(*attributes)
        expected = np.ma.repeat((140, 0, 140, 0), (420, 120, 520, 220))
        expected[expected == 0] = np.ma.masked
        ma_test.assert_masked_array_equal(node.array, expected)

    @patch('analysis_engine.library.at')
    def test_derive__flap_without_weight__standard(self, at):
        mapping = {f: str(f) for f in (0, 17.5, 35)}
        flap_lever = M('Flap Lever', np.ma.repeat(17.5, 1280), values_mapping=mapping)

        attributes = self.generate_attributes('beechcraft')
        at.get_vspeed_map.return_value = self.VSF0

        node = self.node_class()
        node.derive(flap_lever, None, self.airspeed, None, self.liftoffs,
                    self.climbs, *attributes)

        attributes = (a.value for a in attributes)
        at.get_vspeed_map.assert_called_once_with(*attributes)
        expected = np.ma.repeat(0, 1280)
        expected[expected == 0] = np.ma.masked
        ma_test.assert_masked_array_equal(node.array, expected)

    @patch('analysis_engine.library.at')
    def test_derive__flap_without_weight__fallback(self, at):
        mapping = {f: str(f) for f in (0, 17.5, 35)}
        flap_lever = M('Flap Lever', np.ma.repeat(17.5, 1280), values_mapping=mapping)

        attributes = self.generate_attributes('beechcraft')
        at.get_vspeed_map.return_value = self.VSF1

        node = self.node_class()
        node.derive(flap_lever, None, self.airspeed, None, self.liftoffs,
                    self.climbs, *attributes)

        attributes = (a.value for a in attributes)
        at.get_vspeed_map.assert_called_once_with(*attributes)
        expected = np.ma.repeat((135, 0, 135, 0), (420, 120, 520, 220))
        expected[expected == 0] = np.ma.masked
        ma_test.assert_masked_array_equal(node.array, expected)


########################################
# Reference Speed (Vref)


class TestVref(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = Vref
        self.airspeed = P('Airspeed', np.ma.repeat(200, 128))
        self.approaches = buildsections('Approach And Landing', (2, 42), (66, 106))

    def test_can_operate(self):
        # AFR:
        self.assertTrue(self.node_class.can_operate(
            ('Airspeed', 'AFR Vref', 'Approach And Landing'),
            afr_vref=A('AFR Vref', 120),
        ))
        self.assertFalse(self.node_class.can_operate(
            ('Airspeed', 'AFR Vref', 'Approach And Landing'),
            afr_vref=A('AFR Vref', 70),
        ))
        # Embraer:
        self.assertTrue(self.node_class.can_operate(
            ('Airspeed', 'V1-Vref', 'Approach And Landing'),
        ))

    def test_derive__nothing(self):
        node = self.node_class()
        node.derive(self.airspeed, None, None, self.approaches)
        expected = np.ma.repeat(0, 128)
        expected[expected == 0] = np.ma.masked
        ma_test.assert_masked_array_equal(node.array, expected)

    def test_derive__afr_vref(self):
        afr_vref = A('AFR Vref', 120)
        node = self.node_class()
        node.derive(self.airspeed, None, afr_vref, self.approaches)
        expected = np.ma.repeat((0, 120, 0, 120, 0), (2, 40, 24, 40, 22))
        expected[expected == 0] = np.ma.masked
        ma_test.assert_masked_array_equal(node.array, expected)

    def test_derive__embraer(self):
        v1_vref = P('V1-Vref', np.ma.repeat(150, 128))
        node = self.node_class()
        node.derive(self.airspeed, v1_vref, None, self.approaches)
        expected = np.ma.repeat((0, 150, 0, 150, 0), (2, 40, 24, 40, 22))
        expected[expected == 0] = np.ma.masked
        ma_test.assert_masked_array_equal(node.array, expected)


class TestVrefLookup(unittest.TestCase, NodeTest):

    class VSX(VelocitySpeed):
        '''
        Table for aircraft with undefined Vref.
        '''
        tables = {}

    class VSC0(VelocitySpeed):
        '''
        Table for aircraft with configuration and fallback tables.
        '''
        weight_unit = ut.TONNE
        tables = {'vref': {
            'weight': ( 35,  40,  45,  50,  55,  60,  65),
        'Lever Full': (113, 119, 126, 132, 139, 145, 152),
        }}
        fallback = {'vref': {'Lever 3': 140}}

    class VSC1(VelocitySpeed):
        '''
        Table for aircraft with configuration and fallback tables only.
        '''
        weight_unit = None
        fallback = {'vref': {'Lever Full': 135}}

    class VSF0(VelocitySpeed):
        '''
        Table for aircraft with flap and fallback tables.
        '''
        weight_unit = ut.TONNE
        tables = {'vref': {
            'weight': ( 35,  40,  45,  50,  55,  60,  65),
                '40': (113, 119, 126, 132, 139, 145, 152),
        }}
        fallback = {'vref': {'30': 140}}

    class VSF1(VelocitySpeed):
        '''
        Table for aircraft with flap and fallback tables only.
        '''
        weight_unit = None
        fallback = {'vref': {'35': 135}}

    def setUp(self):
        self.node_class = VrefLookup
        self.airspeed = P('Airspeed', np.ma.repeat(200, 128))
        self.weight = P('Gross Weight Smoothed', np.ma.repeat((54192.06, 44192.06), 64))
        self.touchdowns = KTI(name='Touchdown', items=[
            KeyTimeInstance(name='Touchdown', index=7),
            KeyTimeInstance(name='Touchdown', index=71),
        ])
        self.approaches = buildsections('Approach And Landing', (2, 42), (66, 106))

    @patch('analysis_engine.library.at')
    def test_can_operate(self, at):
        nodes = ('Airspeed', 'Approach And Landing', 'Touchdown',
                 'Model', 'Series', 'Family', 'Engine Series', 'Engine Type')
        keys = ('model', 'series', 'family', 'engine_type', 'engine_series')
        airbus = dict(izip(keys, self.generate_attributes('airbus')))
        boeing = dict(izip(keys, self.generate_attributes('boeing')))
        beechcraft = dict(izip(keys, self.generate_attributes('beechcraft')))
        # Assume that lookup tables are found correctly...
        at.get_vspeed_map.return_value = self.VSF0
        # Flap Lever w/ Weight:
        available = nodes + ('Flap Lever', 'Gross Weight Smoothed')
        self.assertTrue(self.node_class.can_operate(available, **boeing))
        # Flap Lever (Synthetic) w/ Weight:
        available = nodes + ('Flap Lever (Synthetic)', 'Gross Weight Smoothed')
        self.assertTrue(self.node_class.can_operate(available, **airbus))
        # Flap Lever w/o Weight:
        available = nodes + ('Flap Lever',)
        self.assertTrue(self.node_class.can_operate(available, **beechcraft))
        # Flap Lever (Synthetic) w/o Weight:
        available = nodes + ('Flap Lever (Synthetic)',)
        self.assertTrue(self.node_class.can_operate(available, **airbus))
        # Assume that lookup tables are not found correctly...
        at.get_vspeed_map.side_effect = (KeyError, self.VSX)
        available = nodes + ('Flap Lever', 'Gross Weight Smoothed')
        for i in range(2):
            self.assertFalse(self.node_class.can_operate(available, **boeing))

    @patch('analysis_engine.library.at')
    def test_derive__flap_with_weight__standard(self, at):
        mapping = {f: str(f) for f in (0, 1, 2, 5, 10, 15, 25, 30, 40)}
        flap_lever = M('Flap Lever', np.ma.repeat(40, 128), values_mapping=mapping)

        attributes = self.generate_attributes('boeing')
        at.get_vspeed_map.return_value = self.VSF0

        node = self.node_class()
        node.derive(flap_lever, None, self.airspeed, self.weight,
                    self.approaches, self.touchdowns, *attributes)

        attributes = (a.value for a in attributes)
        at.get_vspeed_map.assert_called_once_with(*attributes)
        expected = np.ma.repeat((0, 138, 0, 125, 0), (2, 40, 24, 40, 22))
        expected[expected == 0] = np.ma.masked
        ma_test.assert_masked_array_equal(node.array, expected)

    @patch('analysis_engine.library.at')
    def test_derive__flap_with_weight__fallback(self, at):
        mapping = {f: str(f) for f in (0, 1, 2, 5, 10, 15, 25, 30, 40)}
        flap_lever = M('Flap Lever', np.ma.repeat(30, 128), values_mapping=mapping)

        attributes = self.generate_attributes('boeing')
        at.get_vspeed_map.return_value = self.VSF0

        node = self.node_class()
        node.derive(flap_lever, None, self.airspeed, self.weight,
                    self.approaches, self.touchdowns, *attributes)

        attributes = (a.value for a in attributes)
        at.get_vspeed_map.assert_called_once_with(*attributes)
        expected = np.ma.repeat((0, 140, 0, 140, 0), (2, 40, 24, 40, 22))
        expected[expected == 0] = np.ma.masked
        ma_test.assert_masked_array_equal(node.array, expected)

    @patch('analysis_engine.library.at')
    def test_derive__flap_without_weight__standard(self, at):
        mapping = {f: str(f) for f in (0, 17.5, 35)}
        flap_lever = M('Flap Lever', np.ma.repeat(35, 128), values_mapping=mapping)

        attributes = self.generate_attributes('beechcraft')
        at.get_vspeed_map.return_value = self.VSF0

        node = self.node_class()
        node.derive(flap_lever, None, self.airspeed, None, self.approaches,
                    self.touchdowns, *attributes)

        attributes = (a.value for a in attributes)
        at.get_vspeed_map.assert_called_once_with(*attributes)
        expected = np.ma.repeat(0, 128)
        expected[expected == 0] = np.ma.masked
        ma_test.assert_masked_array_equal(node.array, expected)

    @patch('analysis_engine.library.at')
    def test_derive__flap_without_weight__fallback(self, at):
        mapping = {f: str(f) for f in (0, 17.5, 35)}
        flap_lever = M('Flap Lever', np.ma.repeat(35, 128), values_mapping=mapping)

        attributes = self.generate_attributes('beechcraft')
        at.get_vspeed_map.return_value = self.VSF1

        node = self.node_class()
        node.derive(flap_lever, None, self.airspeed, None, self.approaches,
                    self.touchdowns, *attributes)

        attributes = (a.value for a in attributes)
        at.get_vspeed_map.assert_called_once_with(*attributes)
        expected = np.ma.repeat((0, 135, 0, 135, 0), (2, 40, 24, 40, 22))
        expected[expected == 0] = np.ma.masked
        ma_test.assert_masked_array_equal(node.array, expected)


########################################
# Approach Speed (VAPP)


class TestVapp(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = Vapp
        self.airspeed = P('Airspeed', np.ma.repeat(200, 128))
        self.approaches = buildsections('Approach And Landing', (2, 42), (66, 106))

    def test_can_operate(self):
        # AFR:
        self.assertTrue(self.node_class.can_operate(
            ('Airspeed', 'AFR Vapp', 'Approach And Landing'),
            afr_vapp=A('AFR Vapp', 120),
        ))
        self.assertFalse(self.node_class.can_operate(
            ('Airspeed', 'AFR Vapp', 'Approach And Landing'),
            afr_vapp=A('AFR Vapp', 70),
        ))
        # Embraer:
        self.assertTrue(self.node_class.can_operate(
            ('Airspeed', 'VR-Vapp', 'Approach And Landing'),
        ))

    def test_derive__nothing(self):
        node = self.node_class()
        node.derive(self.airspeed, None, None, self.approaches)
        expected = np.ma.repeat(0, 128)
        expected[expected == 0] = np.ma.masked
        ma_test.assert_masked_array_equal(node.array, expected)

    def test_derive__afr_vapp(self):
        afr_vapp = A('AFR Vapp', 120)
        node = self.node_class()
        node.derive(self.airspeed, None, afr_vapp, self.approaches)
        expected = np.ma.repeat((0, 120, 0, 120, 0), (2, 40, 24, 40, 22))
        expected[expected == 0] = np.ma.masked
        ma_test.assert_masked_array_equal(node.array, expected)

    def test_derive__embraer(self):
        vr_vapp = P('VR-Vapp', np.ma.repeat(150, 128))
        node = self.node_class()
        node.derive(self.airspeed, vr_vapp, None, self.approaches)
        expected = np.ma.repeat((0, 150, 0, 150, 0), (2, 40, 24, 40, 22))
        expected[expected == 0] = np.ma.masked
        ma_test.assert_masked_array_equal(node.array, expected)


class TestVappLookup(unittest.TestCase, NodeTest):

    class VSX(VelocitySpeed):
        '''
        Table for aircraft with undefined Vapp.
        '''
        tables = {}

    class VSC0(VelocitySpeed):
        '''
        Table for aircraft with configuration and fallback tables.
        '''
        weight_unit = ut.TONNE
        tables = {'vapp': {
            'weight': ( 35,  40,  45,  50,  55,  60,  65),
        'Lever Full': (113, 119, 126, 132, 139, 145, 152),
        }}
        fallback = {'vapp': {'Lever 3': 140}}

    class VSC1(VelocitySpeed):
        '''
        Table for aircraft with configuration and fallback tables only.
        '''
        weight_unit = None
        fallback = {'vapp': {'Lever Full': 135}}

    class VSF0(VelocitySpeed):
        '''
        Table for aircraft with flap and fallback tables.
        '''
        weight_unit = ut.TONNE
        tables = {'vapp': {
            'weight': ( 35,  40,  45,  50,  55,  60,  65),
                '40': (113, 119, 126, 132, 139, 145, 152),
        }}
        fallback = {'vapp': {'30': 140}}

    class VSF1(VelocitySpeed):
        '''
        Table for aircraft with flap and fallback tables only.
        '''
        weight_unit = None
        fallback = {'vapp': {'35': 135}}

    def setUp(self):
        self.node_class = VappLookup
        self.airspeed = P('Airspeed', np.ma.repeat(200, 128))
        self.weight = P('Gross Weight Smoothed', np.ma.repeat((54192.06, 44192.06), 64))
        self.touchdowns = KTI(name='Touchdown', items=[
            KeyTimeInstance(name='Touchdown', index=7),
            KeyTimeInstance(name='Touchdown', index=71),
        ])
        self.approaches = buildsections('Approach And Landing', (2, 42), (66, 106))

    @patch('analysis_engine.library.at')
    def test_can_operate(self, at):
        nodes = ('Airspeed', 'Approach And Landing', 'Touchdown',
                 'Model', 'Series', 'Family', 'Engine Series', 'Engine Type')
        keys = ('model', 'series', 'family', 'engine_type', 'engine_series')
        airbus = dict(izip(keys, self.generate_attributes('airbus')))
        boeing = dict(izip(keys, self.generate_attributes('boeing')))
        beechcraft = dict(izip(keys, self.generate_attributes('beechcraft')))
        # Assume that lookup tables are found correctly...
        at.get_vspeed_map.return_value = self.VSF0
        # Flap Lever w/ Weight:
        available = nodes + ('Flap Lever', 'Gross Weight Smoothed')
        self.assertTrue(self.node_class.can_operate(available, **boeing))
        # Flap Lever (Synthetic) w/ Weight:
        available = nodes + ('Flap Lever (Synthetic)', 'Gross Weight Smoothed')
        self.assertTrue(self.node_class.can_operate(available, **airbus))
        # Flap Lever w/o Weight:
        available = nodes + ('Flap Lever',)
        self.assertTrue(self.node_class.can_operate(available, **beechcraft))
        # Flap Lever (Synthetic) w/o Weight:
        available = nodes + ('Flap Lever (Synthetic)',)
        self.assertTrue(self.node_class.can_operate(available, **airbus))
        # Assume that lookup tables are not found correctly...
        at.get_vspeed_map.side_effect = (KeyError, self.VSX)
        available = nodes + ('Flap Lever', 'Gross Weight Smoothed')
        for i in range(2):
            self.assertFalse(self.node_class.can_operate(available, **boeing))

    @patch('analysis_engine.library.at')
    def test_derive__flap_with_weight__standard(self, at):
        mapping = {f: str(f) for f in (0, 1, 2, 5, 10, 15, 25, 30, 40)}
        flap_lever = M('Flap Lever', np.ma.repeat(40, 128), values_mapping=mapping)

        attributes = self.generate_attributes('boeing')
        at.get_vspeed_map.return_value = self.VSF0

        node = self.node_class()
        node.derive(flap_lever, None, self.airspeed, self.weight,
                    self.approaches, self.touchdowns, *attributes)

        attributes = (a.value for a in attributes)
        at.get_vspeed_map.assert_called_once_with(*attributes)
        expected = np.ma.repeat((0, 138, 0, 125, 0), (2, 40, 24, 40, 22))
        expected[expected == 0] = np.ma.masked
        ma_test.assert_masked_array_equal(node.array, expected)

    @patch('analysis_engine.library.at')
    def test_derive__flap_with_weight__fallback(self, at):
        mapping = {f: str(f) for f in (0, 1, 2, 5, 10, 15, 25, 30, 40)}
        flap_lever = M('Flap Lever', np.ma.repeat(30, 128), values_mapping=mapping)

        attributes = self.generate_attributes('boeing')
        at.get_vspeed_map.return_value = self.VSF0

        node = self.node_class()
        node.derive(flap_lever, None, self.airspeed, self.weight,
                    self.approaches, self.touchdowns, *attributes)

        attributes = (a.value for a in attributes)
        at.get_vspeed_map.assert_called_once_with(*attributes)
        expected = np.ma.repeat((0, 140, 0, 140, 0), (2, 40, 24, 40, 22))
        expected[expected == 0] = np.ma.masked
        ma_test.assert_masked_array_equal(node.array, expected)

    @patch('analysis_engine.library.at')
    def test_derive__flap_without_weight__standard(self, at):
        mapping = {f: str(f) for f in (0, 17.5, 35)}
        flap_lever = M('Flap Lever', np.ma.repeat(35, 128), values_mapping=mapping)

        attributes = self.generate_attributes('beechcraft')
        at.get_vspeed_map.return_value = self.VSF0

        node = self.node_class()
        node.derive(flap_lever, None, self.airspeed, None, self.approaches,
                    self.touchdowns, *attributes)

        attributes = (a.value for a in attributes)
        at.get_vspeed_map.assert_called_once_with(*attributes)
        expected = np.ma.repeat(0, 128)
        expected[expected == 0] = np.ma.masked
        ma_test.assert_masked_array_equal(node.array, expected)

    @patch('analysis_engine.library.at')
    def test_derive__flap_without_weight__fallback(self, at):
        mapping = {f: str(f) for f in (0, 17.5, 35)}
        flap_lever = M('Flap Lever', np.ma.repeat(35, 128), values_mapping=mapping)

        attributes = self.generate_attributes('beechcraft')
        at.get_vspeed_map.return_value = self.VSF1

        node = self.node_class()
        node.derive(flap_lever, None, self.airspeed, None, self.approaches,
                    self.touchdowns, *attributes)

        attributes = (a.value for a in attributes)
        at.get_vspeed_map.assert_called_once_with(*attributes)
        expected = np.ma.repeat((0, 135, 0, 135, 0), (2, 40, 24, 40, 22))
        expected[expected == 0] = np.ma.masked
        ma_test.assert_masked_array_equal(node.array, expected)


########################################
# Maximum Operating Speed (VMO)


class TestVMOLookup(unittest.TestCase, NodeTest):

    class VSX(VelocitySpeed):
        '''
        Table for aircraft with undefined VMO.
        '''
        tables = {}

    class VS0(VelocitySpeed):
        '''
        Table for aircraft that don't have a VMO.
        '''
        tables = {'vmo': None}

    class VS1(VelocitySpeed):
        '''
        Table for aircraft that have a fixed value for VMO.
        '''
        tables = {'vmo': 350}

    class VS2(VelocitySpeed):
        '''
        Table for aircraft that have linear interpolated values for VMO.
        '''
        tables = {'vmo': {
            'altitude': (  0, 15000, 25000, 40000),
               'speed': (350,   350,   300,   300),
        }}

    class VS3(VelocitySpeed):
        '''
        Table for aircraft that have stepped values for VMO.
        '''
        tables = {'vmo': {
            'altitude': (  0, 20000, 20000, 40000),
               'speed': (350,   350,   300,   300),
        }}

    def setUp(self):
        self.node_class = VMOLookup
        self.altitude = P('Altitude', np.ma.arange(0, 50000, 1000))

    @patch('analysis_engine.library.at')
    def test_can_operate(self, at):
        nodes = ('Altitude STD', 'Model', 'Series', 'Family', 'Engine Series', 'Engine Type')
        keys = ('model', 'series', 'family', 'engine_type', 'engine_series')
        boeing = dict(izip(keys, self.generate_attributes('boeing')))
        # Assume that lookup tables are found correctly...
        at.get_vspeed_map.return_value = self.VS0
        self.assertTrue(self.node_class.can_operate(nodes, **boeing))
        # Assume that lookup tables are not found correctly...
        at.get_vspeed_map.side_effect = (KeyError, self.VSX)
        for i in range(2):
            self.assertFalse(self.node_class.can_operate(nodes, **boeing))

    @patch('analysis_engine.library.at')
    def test_derive__none(self, at):
        attributes = self.generate_attributes('boeing')
        at.get_vspeed_map.return_value = self.VS0

        node = self.node_class()
        node.derive(self.altitude, *attributes)

        attributes = (a.value for a in attributes)
        at.get_vspeed_map.assert_called_once_with(*attributes)
        expected = np.ma.repeat(0, 50)
        expected[expected == 0] = np.ma.masked
        ma_test.assert_masked_array_equal(node.array, expected)

    @patch('analysis_engine.library.at')
    def test_derive__fixed(self, at):
        attributes = self.generate_attributes('boeing')
        at.get_vspeed_map.return_value = self.VS1

        node = self.node_class()
        node.derive(self.altitude, *attributes)

        attributes = (a.value for a in attributes)
        at.get_vspeed_map.assert_called_once_with(*attributes)
        expected = np.ma.repeat(350, 50)
        ma_test.assert_masked_array_equal(node.array, expected)

    @patch('analysis_engine.library.at')
    def test_derive__linear(self, at):
        attributes = self.generate_attributes('boeing')
        at.get_vspeed_map.return_value = self.VS2

        node = self.node_class()
        node.derive(self.altitude, *attributes)

        attributes = (a.value for a in attributes)
        at.get_vspeed_map.assert_called_once_with(*attributes)
        expected = np.ma.concatenate((
            np.ma.repeat(350, 16),
            np.ma.arange(345, 295, -5),
            np.ma.repeat(300, 15),
            np.ma.repeat(000, 9),
        ))
        expected[expected == 0] = np.ma.masked
        ma_test.assert_masked_array_equal(node.array, expected)

    @patch('analysis_engine.library.at')
    def test_derive__stepped(self, at):
        attributes = self.generate_attributes('boeing')
        at.get_vspeed_map.return_value = self.VS3

        node = self.node_class()
        node.derive(self.altitude, *attributes)

        attributes = (a.value for a in attributes)
        at.get_vspeed_map.assert_called_once_with(*attributes)
        expected = np.ma.concatenate((
            np.ma.repeat(350, 20),
            np.ma.repeat(300, 21),
            np.ma.repeat(000, 9),
        ))
        expected[expected == 0] = np.ma.masked
        ma_test.assert_masked_array_equal(node.array, expected)


########################################
# Maximum Operating Mach (MMO)


class TestMMOLookup(unittest.TestCase, NodeTest):

    class VSX(VelocitySpeed):
        '''
        Table for aircraft with undefined MMO.
        '''
        tables = {}

    class VS0(VelocitySpeed):
        '''
        Table for aircraft that don't have a MMO.
        '''
        tables = {'mmo': None}

    class VS1(VelocitySpeed):
        '''
        Table for aircraft that have a fixed value for MMO.
        '''
        tables = {'mmo': 0.850}

    class VS2(VelocitySpeed):
        '''
        Table for aircraft that have linear interpolated values for MMO.
        '''
        tables = {'mmo': {
            'altitude': (    0, 15000, 25000, 40000),
               'speed': (0.850, 0.850, 0.800, 0.800),
        }}

    class VS3(VelocitySpeed):
        '''
        Table for aircraft that have stepped values for MMO.
        '''
        tables = {'mmo': {
            'altitude': (    0, 20000, 20000, 40000),
               'speed': (0.850, 0.850, 0.800, 0.800),
        }}


    def setUp(self):
        self.node_class = MMOLookup
        self.altitude = P('Altitude', np.ma.arange(0, 50000, 1000))

    @patch('analysis_engine.library.at')
    def test_can_operate(self, at):
        nodes = ('Altitude STD', 'Model', 'Series', 'Family', 'Engine Series', 'Engine Type')
        keys = ('model', 'series', 'family', 'engine_type', 'engine_series')
        boeing = dict(izip(keys, self.generate_attributes('boeing')))
        # Assume that lookup tables are found correctly...
        at.get_vspeed_map.return_value = self.VS0
        self.assertTrue(self.node_class.can_operate(nodes, **boeing))
        # Assume that lookup tables are not found correctly...
        at.get_vspeed_map.side_effect = (KeyError, self.VSX)
        for i in range(2):
            self.assertFalse(self.node_class.can_operate(nodes, **boeing))

    @patch('analysis_engine.library.at')
    def test_derive__none(self, at):
        attributes = self.generate_attributes('boeing')
        at.get_vspeed_map.return_value = self.VS0

        node = self.node_class()
        node.derive(self.altitude, *attributes)

        attributes = (a.value for a in attributes)
        at.get_vspeed_map.assert_called_once_with(*attributes)
        expected = np.ma.repeat(0, 50)
        expected[expected == 0] = np.ma.masked
        ma_test.assert_masked_array_equal(node.array, expected)

    @patch('analysis_engine.library.at')
    def test_derive__fixed(self, at):
        attributes = self.generate_attributes('boeing')
        at.get_vspeed_map.return_value = self.VS1

        node = self.node_class()
        node.derive(self.altitude, *attributes)

        attributes = (a.value for a in attributes)
        at.get_vspeed_map.assert_called_once_with(*attributes)
        expected = np.ma.repeat(0.850, 50)
        ma_test.assert_masked_array_equal(node.array, expected)

    @patch('analysis_engine.library.at')
    def test_derive__linear(self, at):
        attributes = self.generate_attributes('boeing')
        at.get_vspeed_map.return_value = self.VS2

        node = self.node_class()
        node.derive(self.altitude, *attributes)

        attributes = (a.value for a in attributes)
        at.get_vspeed_map.assert_called_once_with(*attributes)
        expected = np.ma.concatenate((
            np.ma.repeat(0.850, 16),
            np.ma.arange(0.845, 0.795, -0.005),
            np.ma.repeat(0.800, 15),
            np.ma.repeat(0.000, 9),
        ))
        expected[expected == 0] = np.ma.masked
        ma_test.assert_masked_array_almost_equal(node.array, expected, decimal=3)

    @patch('analysis_engine.library.at')
    def test_derive__stepped(self, at):
        attributes = self.generate_attributes('boeing')
        at.get_vspeed_map.return_value = self.VS3

        node = self.node_class()
        node.derive(self.altitude, *attributes)

        attributes = (a.value for a in attributes)
        at.get_vspeed_map.assert_called_once_with(*attributes)
        expected = np.ma.concatenate((
            np.ma.repeat(0.850, 20),
            np.ma.repeat(0.800, 21),
            np.ma.repeat(0.000, 9),
        ))
        expected[expected == 0] = np.ma.masked
        ma_test.assert_masked_array_equal(node.array, expected)


########################################
# Minimum Airspeed


class TestMinimumAirspeed(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = MinimumAirspeed
        self.operational_combinations = [
            ('Airborne', 'Airspeed', 'VLS'),
            ('Airborne', 'Airspeed', 'Min Operating Speed'),
            ('Airborne', 'Airspeed', 'FC Min Operating Speed'),
            ('Airborne', 'Airspeed', 'FMF Min Manoeuvre Speed'),
            ('Airborne', 'Airspeed', 'FMC Min Manoeuvre Speed', 'Flap Lever'),
            ('Airborne', 'Airspeed', 'FMC Min Manoeuvre Speed', 'Flap Lever (Synthetic)'),
        ]
        self.airspeed = P('Airspeed', np.ma.repeat(250, 100))
        self.airborne = buildsection('Airborne', 10, 90)

    def test_derive__invalid(self):
        node = self.node_class()
        node.derive(self.airspeed, None, None, None, None, None, None, None, self.airborne)
        expected = np.ma.repeat(0, 100)
        expected.mask = True
        ma_test.assert_masked_array_equal(node.array, expected)

    def test_derive__vls(self):
        vls = P('VLS', np.ma.repeat(180, 100))
        node = self.node_class()
        node.derive(self.airspeed, None, None, None, None, vls, None, None, self.airborne)
        expected = np.ma.array(vls.array)
        expected.mask = np.repeat((1, 0, 0, 0, 0, 0, 0, 0, 0, 1), 10)
        ma_test.assert_masked_array_equal(node.array, expected)

    def test_derive__mos(self):
        mos = P('Min Operating Speed', np.ma.repeat(190, 100))
        node = self.node_class()
        node.derive(self.airspeed, None, None, None, mos, None, None, None, self.airborne)
        expected = np.ma.array(mos.array)
        expected.mask = np.repeat((1, 0, 0, 0, 0, 0, 0, 0, 0, 1), 10)
        ma_test.assert_masked_array_equal(node.array, expected)

    def test_derive__mos_fc(self):
        mos_fc = P('FC Min Operating Speed', np.ma.repeat(200, 100))
        node = self.node_class()
        node.derive(self.airspeed, None, None, mos_fc, None, None, None, None, self.airborne)
        expected = np.ma.array(mos_fc.array)
        expected.mask = np.repeat((1, 0, 0, 0, 0, 0, 0, 0, 0, 1), 10)
        ma_test.assert_masked_array_equal(node.array, expected)

    def test_derive__mms_fmc(self):
        mms_fmc = P('FMC Min Manoeuvre Speed', np.ma.repeat(220, 100))
        array = np.ma.repeat((20, 10, 10, 0, 0, 0, 0, 10, 10, 20), 10)
        flap = M('Flap Lever', array, values_mapping={0: '0', 10: '10', 20: '20'})
        node = self.node_class()
        node.derive(self.airspeed, None, mms_fmc, None, None, None, flap, None, self.airborne)
        expected = np.ma.array(mms_fmc.array)
        expected.mask = np.repeat((1, 1, 1, 0, 0, 0, 0, 1, 1, 1), 10)
        ma_test.assert_masked_array_equal(node.array, expected)

    def test_derive__mms_fmf(self):
        mms_fmf = P('FMF Min Manoeuvre Speed', np.ma.repeat(210, 100))
        node = self.node_class()
        node.derive(self.airspeed, mms_fmf, None, None, None, None, None, None, self.airborne)
        expected = np.ma.array(mms_fmf.array)
        expected.mask = np.repeat((1, 0, 0, 0, 0, 0, 0, 0, 0, 1), 10)
        ma_test.assert_masked_array_equal(node.array, expected)


########################################
# Flap Manoeuvre Speed


class TestFlapManoeuvreSpeed(unittest.TestCase, NodeTest):

    class VSX(VelocitySpeed):
        '''
        Table for aircraft with undefined Vref.
        '''
        tables = {}

    class VS0(VelocitySpeed):
        '''
        Table for aircraft with flap and weight.
        '''
        weight_unit = ut.TONNE
        tables = {'vref': {
            'weight': ( 35,  40,  45,  50,  55,  60,  65),
                '30': (111, 119, 127, 134, 141, 147, 154),
                '40': (107, 115, 123, 131, 138, 146, 153),
        },
        'vmo': 340,
        'mmo': 0.82,
    }

    def setUp(self):
        self.node_class = FlapManoeuvreSpeed
        self.airspeed = P('Airspeed', np.ma.repeat(200, 90))
        self.weight = P('Gross Weight Smoothed', np.ma.repeat((50000, 60000), 45))
        self.altitude = P('Altitude STD', np.ma.arange(20010, 19920, -1))
        self.descents = buildsections('Descent To Flare', (2, 42), (47, 87))
        self.flap_lever = M(
            name='Flap Lever',
            array=np.ma.repeat((0, 1, 2, 5, 10, 15, 25, 30, 40), 10),
            values_mapping={f: str(f) for f in (0, 1, 2, 5, 10, 15, 25, 30, 40)},
        )
        self.flap_lever.array.mask = np.repeat(False, 90)  # expand mask.

    @patch('analysis_engine.derived_parameters.at')
    @patch('analysis_engine.library.at')
    def test_can_operate(self, at0, at1):
        nodes = ('Airspeed', 'Altitude STD', 'Descent To Flare',
                 'Gross Weight Smoothed', 'Model', 'Series', 'Family',
                 'Engine Series', 'Engine Type')
        attrs = dict(izip(
            ('model', 'series', 'family', 'engine_type', 'engine_series'),
            self.generate_attributes('boeing'),
        ))
        attrs.update(manufacturer=A('Manufacturer', 'Boeing'))
        # Assume that lookup tables are found correctly...
        at0.get_vspeed_map.return_value = self.VS0
        at1.get_fms_map.return_value = {}
        # Flap Lever:
        available = nodes + ('Flap Lever',)
        self.assertTrue(self.node_class.can_operate(available, **attrs))
        # Flap Lever (Synthetic):
        available = nodes + ('Flap Lever (Synthetic)',)
        self.assertTrue(self.node_class.can_operate(available, **attrs))
        # Recorded Vref:
        available = nodes + ('Flap Lever', 'Vref (25)', 'Vref (30)')
        self.assertTrue(self.node_class.can_operate(available, **attrs))
        # Assume that lookup tables are not found correctly...
        # Please be careful if changing the below side effect iterables!
        at0.get_vspeed_map.side_effect = (KeyError, self.VSX)
        at1.get_fms_map.side_effect = ({}, {}, KeyError)
        available = nodes + ('Flap Lever',)
        for i in range(3):
            self.assertFalse(self.node_class.can_operate(available, **attrs))

    @patch('analysis_engine.derived_parameters.at')
    @patch('analysis_engine.library.at')
    def test_derive__vref_plus_offset(self, at0, at1):
        attributes = self.generate_attributes('boeing')
        at0.get_vspeed_map.return_value = self.VS0
        at1.get_fms_map.return_value = {
            '0':  ('40', 70),
            '1':  ('40', 50),
            '2':  None,
            '5':  ('40', 30),
            '10': ('40', 30),
            '15': ('40', 20),
            '25': ('40', 10),
            '30': ('30', 0),
            '40': ('40', 0),
        }

        node = self.node_class()
        node.derive(self.airspeed, self.flap_lever, None, self.weight,
                    None, None, self.altitude, self.descents, *attributes)

        attributes = [a.value for a in attributes]
        at0.get_vspeed_map.assert_called_once_with(*attributes)
        at1.get_fms_map.assert_called_once_with(*attributes[:3])
        expected = np.ma.repeat(
            (201, 181, 0, 161, 161, 176, 166, 156, 147, 146),
            (10, 10, 10, 10, 5, 5, 10, 10, 10, 10)
        )
        for s in slice(0, 10), slice(20, 30), slice(42, 47), slice(87, 90):
            expected[s] = np.ma.masked
        ma_test.assert_masked_array_almost_equal(node.array, expected, decimal=0)

    @patch('analysis_engine.derived_parameters.at')
    @patch('analysis_engine.library.at')
    def test_derive__fixed_speeds_in_weight_bands(self, at0, at1):
        attributes = self.generate_attributes('boeing')
        at0.get_vspeed_map.return_value = self.VS0
        at1.get_fms_map.return_value = {
            '0':  ((53070, 210), (62823, 220), (99999, 230)),
            '1':  ((53070, 190), (62823, 200), (99999, 210)),
            '2':  None,
            '5':  ((53070, 170), (62823, 180), (99999, 190)),
            '10': ((53070, 160), (62823, 170), (99999, 180)),
            '15': ((53070, 150), (62823, 160), (99999, 170)),
            '25': ((53070, 140), (62823, 150), (99999, 160)),
            '30': ('30', 0),
            '40': ('40', 0),
        }

        node = self.node_class()
        node.derive(self.airspeed, self.flap_lever, None, self.weight,
                    None, None, self.altitude, self.descents, *attributes)

        attributes = [a.value for a in attributes]
        at0.get_vspeed_map.assert_called_once_with(*attributes)
        at1.get_fms_map.assert_called_once_with(*attributes[:3])
        expected = np.ma.repeat(
            (210, 190, 0, 170, 160, 170, 160, 150, 147, 146),
            (10, 10, 10, 10, 5, 5, 10, 10, 10, 10)
        )
        for s in slice(0, 10), slice(20, 30), slice(42, 47), slice(87, 90):
            expected[s] = np.ma.masked
        ma_test.assert_masked_array_almost_equal(node.array, expected, decimal=0)


    @patch('analysis_engine.derived_parameters.at')
    @patch('analysis_engine.library.at')
    def test_derive__use_recorded_vref(self, at0, at1):
        attributes = self.generate_attributes('boeing')
        at0.get_vspeed_map.return_value = self.VS0
        at1.get_fms_map.return_value = {
            '0':  ('30', 80),
            '1':  ('30', 60),
            '5':  ('30', 40),
            '15': ('30', 20),
            '20': ('30', 20),
            '25': ('25', 0),
            '30': ('30', 0),
        }

        # We need to change some of the test data:
        self.airspeed = P('Airspeed', np.ma.repeat(200, 70))
        self.weight = P('Gross Weight Smoothed', np.ma.repeat((50000, 60000), 35))
        self.descents = buildsections('Descent To Flare', (2, 32), (37, 67))
        self.flap_lever = M(
            name='Flap Lever',
            array=np.ma.repeat((0, 1, 5, 15, 20, 25, 30), 10),
            values_mapping={f: str(f) for f in (0, 1, 5, 15, 20, 25, 30)},
        )
        self.flap_lever.array.mask = np.repeat(False, 70)  # expand mask.
        self.vref_25 = P('Vref (25)', np.ma.repeat(155, 70))
        self.vref_30 = P('Vref (30)', np.ma.repeat(150, 70))

        node = self.node_class()
        node.derive(self.airspeed, self.flap_lever, None, self.weight,
                    self.vref_25, self.vref_30, self.altitude, self.descents,
                    *attributes)

        attributes = [a.value for a in attributes]
        at0.get_vspeed_map.assert_called_once_with(*attributes)
        at1.get_fms_map.assert_called_once_with(*attributes[:3])
        expected = np.ma.repeat((230, 210, 190, 170, 170, 155, 150), 10)
        for s in slice(0, 10), slice(32, 37), slice(67, 70):
            expected[s] = np.ma.masked
        ma_test.assert_masked_array_equal(node.array, expected)


##############################################################################
# Relative Airspeeds


########################################
# Airspeed Minus V2


class TestAirspeedMinusV2(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = AirspeedMinusV2
        self.operational_combinations = [
            ('Airspeed', 'V2', 'Liftoff', 'Climb Start'),
            ('Airspeed', 'V2 Lookup', 'Liftoff', 'Climb Start'),
            ('Airspeed', 'V2', 'V2 Lookup', 'Liftoff', 'Climb Start'),
        ]
        self.airspeed = P('Airspeed', np.ma.repeat(102, 2000))
        self.v2_record = P('V2', np.ma.repeat((90, 120), 1000))
        self.v2_lookup = P('V2 Lookup', np.ma.repeat((95, 125), 1000))
        self.liftoffs = KTI(name='Liftoff', items=[
            KeyTimeInstance(name='Liftoff', index=500),
        ])
        self.climbs = KTI(name='Climb Start', items=[
            KeyTimeInstance(name='Climb Start', index=1000),
        ])

    def test_derive__record_only(self):
        node = self.node_class()
        node.derive(self.airspeed, self.v2_record, None, self.liftoffs, self.climbs)
        expected = np.ma.repeat((0, 12, 0), (180, 820, 1000))
        expected[expected == 0] = np.ma.masked
        ma_test.assert_masked_array_equal(node.array, expected)

    def test_derive__lookup_only(self):
        node = self.node_class()
        node.derive(self.airspeed, None, self.v2_lookup, self.liftoffs, self.climbs)
        expected = np.ma.repeat((0, 7, 0), (180, 820, 1000))
        expected[expected == 0] = np.ma.masked
        ma_test.assert_masked_array_equal(node.array, expected)

    def test_derive__prefer_record(self):
        node = self.node_class()
        node.derive(self.airspeed, self.v2_record, self.v2_lookup, self.liftoffs, self.climbs)
        expected = np.ma.repeat((0, 12, 0), (180, 820, 1000))
        expected[expected == 0] = np.ma.masked
        ma_test.assert_masked_array_equal(node.array, expected)

    def test_derive__record_masked(self):
        self.v2_record.array.mask = True
        node = self.node_class()
        node.derive(self.airspeed, self.v2_record, self.v2_lookup, self.liftoffs, self.climbs)
        expected = np.ma.repeat((0, 7, 0), (180, 820, 1000))
        expected[expected == 0] = np.ma.masked
        ma_test.assert_masked_array_equal(node.array, expected)

    def test_derive__both_masked(self):
        self.v2_record.array.mask = True
        self.v2_lookup.array.mask = True
        node = self.node_class()
        node.derive(self.airspeed, self.v2_record, self.v2_lookup, self.liftoffs, self.climbs)
        expected = np.ma.repeat(0, 2000)
        expected[expected == 0] = np.ma.masked
        ma_test.assert_masked_array_equal(node.array, expected)

    def test_derive__masked_within_phase(self):
        self.v2_record.array[:-1] = np.ma.masked
        node = self.node_class()
        node.derive(self.airspeed, self.v2_record, self.v2_lookup, self.liftoffs, self.climbs)
        expected = np.ma.repeat((0, 7, 0), (180, 820, 1000))
        expected[expected == 0] = np.ma.masked
        ma_test.assert_masked_array_equal(node.array, expected)


class TestAirspeedMinusV2For3Sec(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = AirspeedMinusV2For3Sec
        self.operational_combinations = [
            ('Airspeed Minus V2',),
        ]
        self.airspeed = P(
            name='Airspeed Minus V2',
            array=np.ma.repeat((100, 110, 120, 100), (6, 7, 1, 6)),
            frequency=2,
        )

    def test_derive_basic(self):
        node = self.node_class()
        node.get_derived([self.airspeed])
        expected = np.ma.repeat((100, 110, 100), (6, 8, 6))
        expected[-6:] = np.ma.masked
        ma_test.assert_masked_array_equal(node.array, expected)

    def test_derive_align(self):
        self.airspeed.frequency = 1
        node = self.node_class()
        node.get_derived([self.airspeed])
        expected = np.ma.repeat((100, 105, 110, 100), (11, 1, 16, 12))
        expected[-7:] = np.ma.masked
        ma_test.assert_masked_array_equal(node.array, expected)


########################################
# Airspeed Minus Vref


class TestAirspeedMinusVref(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = AirspeedMinusVref
        self.operational_combinations = [
            ('Airspeed', 'Vref', 'Approach And Landing'),
            ('Airspeed', 'Vref Lookup', 'Approach And Landing'),
            ('Airspeed', 'Vref', 'Vref Lookup', 'Approach And Landing'),
        ]
        self.airspeed = P('Airspeed', np.ma.repeat(102, 2000))
        self.vref_record = P('Vref', np.ma.repeat((90, 120), 1000))
        self.vref_lookup = P('Vref Lookup', np.ma.repeat((95, 125), 1000))
        self.approaches = buildsection('Approach And Landing', 500, 1000)

    def test_derive__record_only(self):
        node = self.node_class()
        node.derive(self.airspeed, self.vref_record, None, self.approaches)
        expected = np.ma.repeat((0, 12, 0, 0), 500)
        expected[expected == 0] = np.ma.masked
        ma_test.assert_masked_array_equal(node.array, expected)

    def test_derive__lookup_only(self):
        node = self.node_class()
        node.derive(self.airspeed, None, self.vref_lookup, self.approaches)
        expected = np.ma.repeat((0, 7, 0, 0), 500)
        expected[expected == 0] = np.ma.masked
        ma_test.assert_masked_array_equal(node.array, expected)

    def test_derive__prefer_record(self):
        node = self.node_class()
        node.derive(self.airspeed, self.vref_record, self.vref_lookup, self.approaches)
        expected = np.ma.repeat((0, 12, 0, 0), 500)
        expected[expected == 0] = np.ma.masked
        ma_test.assert_masked_array_equal(node.array, expected)

    def test_derive__record_masked(self):
        self.vref_record.array.mask = True
        node = self.node_class()
        node.derive(self.airspeed, self.vref_record, self.vref_lookup, self.approaches)
        expected = np.ma.repeat((0, 7, 0, 0), 500)
        expected[expected == 0] = np.ma.masked
        ma_test.assert_masked_array_equal(node.array, expected)

    def test_derive__both_masked(self):
        self.vref_record.array.mask = True
        self.vref_lookup.array.mask = True
        node = self.node_class()
        node.derive(self.airspeed, self.vref_record, self.vref_lookup, self.approaches)
        expected = np.ma.repeat(0, 2000)
        expected[expected == 0] = np.ma.masked
        ma_test.assert_masked_array_equal(node.array, expected)

    def test_derive__masked_within_phase(self):
        self.vref_record.array[:-1] = np.ma.masked
        node = self.node_class()
        node.derive(self.airspeed, self.vref_record, self.vref_lookup, self.approaches)
        expected = np.ma.repeat((0, 7, 0, 0), 500)
        expected[expected == 0] = np.ma.masked
        ma_test.assert_masked_array_equal(node.array, expected)

    def test_derive__superframe(self):
        self.vref_record.array = np.tile(np.ma.repeat((100, 100, 100, 120), 4), 2)
        self.vref_record.frequency = 1 / 64.0
        node = self.node_class()
        node.get_derived([self.airspeed, self.vref_record, None, self.approaches])
        expected = np.ma.repeat((0, 2, 0, 0), 500)
        expected[expected == 0] = np.ma.masked
        ma_test.assert_masked_array_equal(node.array, expected)


class TestAirspeedMinusVrefFor3Sec(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = AirspeedMinusVrefFor3Sec
        self.operational_combinations = [
            ('Airspeed Minus Vref',),
        ]
        self.airspeed = P(
            name='Airspeed Minus Vref',
            array=np.ma.repeat((100, 110, 120, 100), (6, 7, 1, 6)),
            frequency=2,
        )

    def test_derive_basic(self):
        node = self.node_class()
        node.get_derived([self.airspeed])
        expected = np.ma.repeat((100, 110, 100), (6, 8, 6))
        expected[-6:] = np.ma.masked
        ma_test.assert_masked_array_equal(node.array, expected)

    def test_derive_align(self):
        self.airspeed.frequency = 1
        node = self.node_class()
        node.get_derived([self.airspeed])
        expected = np.ma.array([100.0] * 11 + [105] + [110] * 16 + [100] * 12)
        expected = np.ma.repeat((100, 105, 110, 100), (11, 1, 16, 12))
        expected[-7:] = np.ma.masked
        ma_test.assert_masked_array_equal(node.array, expected)


########################################
# Airspeed Minus Vapp


class TestAirspeedMinusVapp(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = AirspeedMinusVapp
        self.operational_combinations = [
            ('Airspeed', 'Vapp', 'Approach And Landing'),
            ('Airspeed', 'Vapp Lookup', 'Approach And Landing'),
            ('Airspeed', 'Vapp', 'Vapp Lookup', 'Approach And Landing'),
        ]
        self.airspeed = P('Airspeed', np.ma.repeat(102, 2000))
        self.vapp_record = P('Vapp', np.ma.repeat((90, 120), 1000))
        self.vapp_lookup = P('Vapp Lookup', np.ma.repeat((95, 125), 1000))
        self.approaches = buildsection('Approach And Landing', 500, 1000)

    def test_derive__record_only(self):
        node = self.node_class()
        node.derive(self.airspeed, self.vapp_record, None, self.approaches)
        expected = np.ma.repeat((0, 12, 0, 0), 500)
        expected[expected == 0] = np.ma.masked
        ma_test.assert_masked_array_equal(node.array, expected)

    def test_derive__lookup_only(self):
        node = self.node_class()
        node.derive(self.airspeed, None, self.vapp_lookup, self.approaches)
        expected = np.ma.repeat((0, 7, 0, 0), 500)
        expected[expected == 0] = np.ma.masked
        ma_test.assert_masked_array_equal(node.array, expected)

    def test_derive__prefer_record(self):
        node = self.node_class()
        node.derive(self.airspeed, self.vapp_record, self.vapp_lookup, self.approaches)
        expected = np.ma.repeat((0, 12, 0, 0), 500)
        expected[expected == 0] = np.ma.masked
        ma_test.assert_masked_array_equal(node.array, expected)

    def test_derive__record_masked(self):
        self.vapp_record.array.mask = True
        node = self.node_class()
        node.derive(self.airspeed, self.vapp_record, self.vapp_lookup, self.approaches)
        expected = np.ma.repeat((0, 7, 0, 0), 500)
        expected[expected == 0] = np.ma.masked
        ma_test.assert_masked_array_equal(node.array, expected)

    def test_derive__both_masked(self):
        self.vapp_record.array.mask = True
        self.vapp_lookup.array.mask = True
        node = self.node_class()
        node.derive(self.airspeed, self.vapp_record, self.vapp_lookup, self.approaches)
        expected = np.ma.repeat(0, 2000)
        expected[expected == 0] = np.ma.masked
        ma_test.assert_masked_array_equal(node.array, expected)

    def test_derive__masked_within_phase(self):
        self.vapp_record.array[:-1] = np.ma.masked
        node = self.node_class()
        node.derive(self.airspeed, self.vapp_record, self.vapp_lookup, self.approaches)
        expected = np.ma.repeat((0, 7, 0, 0), 500)
        expected[expected == 0] = np.ma.masked
        ma_test.assert_masked_array_equal(node.array, expected)

    def test_derive__superframe(self):
        self.vapp_record.array = np.tile(np.ma.repeat((100, 100, 100, 120), 4), 2)
        self.vapp_record.frequency = 1 / 64.0
        node = self.node_class()
        node.get_derived([self.airspeed, self.vapp_record, None, self.approaches])
        expected = np.ma.repeat((0, 2, 0, 0), 500)
        expected[expected == 0] = np.ma.masked
        ma_test.assert_masked_array_equal(node.array, expected)


class TestAirspeedMinusVappFor3Sec(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = AirspeedMinusVappFor3Sec
        self.operational_combinations = [
            ('Airspeed Minus Vapp',),
        ]
        self.airspeed = P(
            name='Airspeed Minus Vapp',
            array=np.ma.repeat((100, 110, 120, 100), (6, 7, 1, 6)),
            frequency=2,
        )

    def test_derive_basic(self):
        node = self.node_class()
        node.get_derived([self.airspeed])
        expected = np.ma.repeat((100, 110, 100), (6, 8, 6))
        expected[-6:] = np.ma.masked
        ma_test.assert_masked_array_equal(node.array, expected)

    def test_derive_align(self):
        self.airspeed.frequency = 1
        node = self.node_class()
        node.get_derived([self.airspeed])
        expected = np.ma.repeat((100, 105, 110, 100), (11, 1, 16, 12))
        expected[-7:] = np.ma.masked
        ma_test.assert_masked_array_equal(node.array, expected)


########################################
# Airspeed Minus Minimum Airspeed


class TestAirspeedMinusMinimumAirspeed(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = AirspeedMinusMinimumAirspeed
        self.operational_combinations = [
            ('Airspeed', 'Minimum Airspeed'),
        ]
        self.airspeed = P('Airspeed', np.ma.repeat(102, 6))
        self.minimum_airspeed = P('Minimum Airspeed', np.ma.arange(80, 92, 2))

    def test_derive__basic(self):
        self.minimum_airspeed.array[3] = np.ma.masked
        node = self.node_class()
        node.derive(self.airspeed, self.minimum_airspeed)
        expected = np.ma.arange(22, 10, -2)
        expected[3] = np.ma.masked
        ma_test.assert_masked_array_equal(node.array, expected)

    def test_derive__masked(self):
        self.minimum_airspeed.array.mask = True
        node = self.node_class()
        node.derive(self.airspeed, self.minimum_airspeed)
        expected = np.ma.arange(22, 10, -2)
        expected.mask = True
        ma_test.assert_masked_array_equal(node.array, expected)


class TestAirspeedMinusMinimumAirspeedFor3Sec(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = AirspeedMinusMinimumAirspeedFor3Sec
        self.operational_combinations = [
            ('Airspeed Minus Minimum Airspeed',),
        ]
        self.airspeed = P(
            name='Airspeed Minus Minimum Airspeed',
            array=np.ma.repeat((100, 110, 120, 100), (6, 7, 1, 6)),
            frequency=2,
        )

    def test_derive_basic(self):
        node = self.node_class()
        node.get_derived([self.airspeed])
        expected = np.ma.repeat((100, 110, 100), (6, 8, 6))
        expected[-6:] = np.ma.masked
        ma_test.assert_masked_array_equal(node.array, expected)

    def test_derive_align(self):
        self.airspeed.frequency = 1
        node = self.node_class()
        node.get_derived([self.airspeed])
        expected = np.ma.repeat((100, 105, 110, 100), (11, 1, 16, 12))
        expected[-7:] = np.ma.masked
        ma_test.assert_masked_array_equal(node.array, expected)


########################################
# Airspeed Minus Flap Manoeuvre Speed


class TestAirspeedMinusFlapManoeuvreSpeed(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = AirspeedMinusFlapManoeuvreSpeed
        self.operational_combinations = [
            ('Airspeed', 'Flap Manoeuvre Speed'),
        ]
        self.airspeed = P('Airspeed', np.ma.repeat(102, 6))
        self.flap_mvr_spd = P('Flap Manoeuvre Speed', np.ma.arange(80, 92, 2))

    def test_derive__basic(self):
        self.flap_mvr_spd.array[3] = np.ma.masked
        node = self.node_class()
        node.derive(self.airspeed, self.flap_mvr_spd)
        expected = np.ma.arange(22, 10, -2)
        expected[3] = np.ma.masked
        ma_test.assert_masked_array_equal(node.array, expected)

    def test_derive__masked(self):
        self.flap_mvr_spd.array.mask = True
        node = self.node_class()
        node.derive(self.airspeed, self.flap_mvr_spd)
        expected = np.ma.arange(22, 10, -2)
        expected.mask = True
        ma_test.assert_masked_array_equal(node.array, expected)


class TestAirspeedMinusFlapManoeuvreSpeedFor3Sec(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = AirspeedMinusFlapManoeuvreSpeedFor3Sec
        self.operational_combinations = [
            ('Airspeed Minus Flap Manoeuvre Speed',),
        ]
        self.airspeed = P(
            name='Airspeed Minus Flap Manoeuvre Speed',
            array=np.ma.repeat((100, 110, 120, 100), (6, 7, 1, 6)),
            frequency=2,
        )

    def test_derive_basic(self):
        node = self.node_class()
        node.get_derived([self.airspeed])
        expected = np.ma.repeat((100, 110, 100), (6, 8, 6))
        expected[-6:] = np.ma.masked
        ma_test.assert_masked_array_equal(node.array, expected)

    def test_derive_align(self):
        self.airspeed.frequency = 1
        node = self.node_class()
        node.get_derived([self.airspeed])
        expected = np.ma.repeat((100, 105, 110, 100), (11, 1, 16, 12))
        expected[-7:] = np.ma.masked
        ma_test.assert_masked_array_equal(node.array, expected)


########################################
# Airspeed Relative


class TestAirspeedRelative(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = AirspeedRelative
        self.operational_combinations = [
            ('Airspeed Minus Vapp',),
            ('Airspeed Minus Vref',),
            ('Airspeed Minus Vapp', 'Airspeed Minus Vref'),
        ]
        self.vapp = P('Airspeed Minus Vapp', np.ma.arange(100, 200))
        self.vref = P('Airspeed Minus Vref', np.ma.arange(200, 300))

    def test_derive__vapp(self):
        node = self.node_class()
        node.derive(self.vapp, None)
        ma_test.assert_masked_array_equal(node.array, self.vapp.array)

    def test_derive__vref(self):
        node = self.node_class()
        node.derive(None, self.vref)
        ma_test.assert_masked_array_equal(node.array, self.vref.array)

    def test_derive__both(self):
        node = self.node_class()
        node.derive(self.vapp, self.vref)
        ma_test.assert_masked_array_equal(node.array, self.vapp.array)


class TestAirspeedRelativeFor3Sec(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = AirspeedRelativeFor3Sec
        self.operational_combinations = [
            ('Airspeed Minus Vapp For 3 Sec',),
            ('Airspeed Minus Vref For 3 Sec',),
            ('Airspeed Minus Vapp For 3 Sec', 'Airspeed Minus Vref For 3 Sec'),
        ]
        self.vapp = P('Airspeed Minus Vapp For 3 Sec', np.ma.arange(100, 200), frequency=2)
        self.vref = P('Airspeed Minus Vref For 3 Sec', np.ma.arange(200, 300), frequency=2)

    def test_derive__vapp(self):
        node = self.node_class()
        node.get_derived([self.vapp, None])
        ma_test.assert_masked_array_equal(node.array, self.vapp.array)

    def test_derive__vref(self):
        node = self.node_class()
        node.get_derived([None, self.vref])
        ma_test.assert_masked_array_equal(node.array, self.vref.array)

    def test_derive__both(self):
        node = self.node_class()
        node.get_derived([self.vapp, self.vref])
        ma_test.assert_masked_array_equal(node.array, self.vapp.array)


##############################################################################


if __name__ == '__main__':
    ##suite = unittest.TestSuite()
    ##suite.addTest(TestConfiguration('test_time_taken2'))
    ##unittest.TextTestRunner(verbosity=2).run(suite)
    pass
