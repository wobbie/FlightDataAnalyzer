import datetime
import numpy as np
import os
import unittest

from mock import patch

from hdfaccess.parameter import MappedArray
from flightdatautilities import aircrafttables as at, units as ut
from flightdatautilities.aircrafttables.constants import AVAILABLE_CONF_STATES
from flightdatautilities import masked_array_testutils as ma_test

from analysis_engine.library import (
    unique_values,
)
from analysis_engine.node import (
    Attribute,
    A,
    #App,
    #ApproachItem,
    #KeyPointValue,
    #KPV,
    #KeyTimeInstance,
    #KTI,
    load,
    M,
    P,
    Section,
    S,
)
from analysis_engine.multistate_parameters import (
    APEngaged,
    APChannelsEngaged,
    APLateralMode,
    APVerticalMode,
    APUOn,
    APURunning,
    Configuration,
    Daylight,
    DualInputWarning,
    ThrustModeSelected,
    Eng_1_Fire,
    Eng_2_Fire,
    Eng_3_Fire,
    Eng_4_Fire,
    Eng_Fire,
    Eng_Oil_Press_Warning,
    EventMarker,
    Flap,
    FlapExcludingTransition,
    FlapIncludingTransition,
    FlapLever,
    FlapLeverSynthetic,
    Flaperon,
    FuelQty_Low,
    GearDown,
    GearDownSelected,
    GearInTransit,
    GearOnGround,
    GearUp,
    GearUpSelected,
    Gear_RedWarning,
    KeyVHFCapt,
    KeyVHFFO,
    MasterWarning,
    PackValvesOpen,
    PilotFlying,
    PitchAlternateLaw,
    Slat,
    SlatExcludingTransition,
    SlatIncludingTransition,
    SpeedControl,
    SpeedbrakeDeployed,
    SpeedbrakeSelected,
    StableApproach,
    StickPusher,
    StickShaker,
    TakeoffConfigurationWarning,
    TAWSAlert,
    TAWSDontSink,
    TAWSGlideslopeCancel,
    TAWSTooLowGear,
    TCASFailure,
    ThrustReversers,
)

##############################################################################
# Test Configuration


def setUpModule():
    at.configure(package='flightdatautilities.aircrafttables')


##############################################################################


test_data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              'test_data')

class NodeTest(object):
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
        from hdfaccess.file import hdf_file

        params = []
        phase = None

        with tempfile.NamedTemporaryFile() as temp_file:
            shutil.copy(hdf_path, temp_file.name)

            with hdf_file(hdf_path) as hdf:
                for param_name in param_names:
                    p = hdf.get(param_name)
                    if param_name == 'Pilot Flying':
                        p.array.values_mapping = {0: '-', 1: 'Captain', 2: 'First Officer'}
                    params.append(p)

        if _slice:
            phase = S(name=phase_name, frequency=1)
            phase.create_section(_slice)
            phase = phase.get_aligned(params[0])

        return params, phase



class TestAPLateralMode(unittest.TestCase):
    
    def test_can_operate(self):
        # Avoid exploding long list of combinations.
        self.assertTrue(APLateralMode.can_operate(['Lateral Mode Selected']))
        self.assertTrue(APLateralMode.can_operate(['Runway Mode Active']))
        self.assertTrue(APLateralMode.can_operate([
            'Lateral Mode Selected',
            'Runway Mode Active',
            'NAV Mode Active',
            'ILS Localizer Capture Active',
            'ILS Localizer Track Active',
            'Roll Go Around Mode Active',
            'Land Track Active',
            'Heading Mode Active']))
    
    def test_derive_lateral_mode_selected(self):
        lateral_mode_selected_values_mapping = {
            0: '-',
            1: 'Runway Mode Active',
            2: 'NAV Mode Active',
            3: 'ILS Localizer Capture Active',
            4: 'Unused Mode',
        }
        lateral_mode_selected_array = np.ma.array(
            [0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
        lateral_mode_selected = M(
            'Lateral Mode Selected',
            array=lateral_mode_selected_array,
            values_mapping=lateral_mode_selected_values_mapping)
        node = APLateralMode()
        node.derive(
            lateral_mode_selected, None, None, None, None, None, None, None)
        self.assertTrue(
            all(node.array ==
                ['-', '-', 'RWY', 'RWY', 'NAV', 'NAV', 'LOC CAPT', 'LOC CAPT', '-', '-']))
    
    def test_derive_all(self):
        activated_values_mapping = {0: '-', 1: 'Activated'}
        runway_mode_active = M(
            'Runway Mode Active',
            array=np.ma.array([0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            values_mapping=activated_values_mapping,
        )
        nav_mode_active = M(
            'NAV Mode Active',
            array=np.ma.array([0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            values_mapping=activated_values_mapping,
        )
        ils_localizer_capture_active = M(
            'ILS Localizer Capture Active',
            array=np.ma.array([0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            values_mapping=activated_values_mapping,
        )
        ils_localizer_track_active = M(
            'ILS Localizer Track Active',
            array=np.ma.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]),
            values_mapping=activated_values_mapping,
        )
        roll_go_around_mode_active = M(
            'Roll Go Around Mode Active',
            array=np.ma.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0]),
            values_mapping=activated_values_mapping,
        )
        land_track_active = M(
            'Land Track Active',
            array=np.ma.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]),
            values_mapping=activated_values_mapping,
        )
        heading_mode_active = M(
            'Heading Mode Active',
            array=np.ma.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0]),
            values_mapping=activated_values_mapping,
        )
        node = APLateralMode()
        node.derive(
            None, runway_mode_active, nav_mode_active,
            ils_localizer_capture_active, ils_localizer_track_active,
            roll_go_around_mode_active, land_track_active, heading_mode_active)
        self.assertTrue(
            all(node.array == ['-', '-',
                               'RWY', 'RWY',
                               'NAV', 'NAV',
                               'LOC CAPT', 'LOC CAPT',
                               'LOC', 'LOC',
                               'ROLL OUT', 'ROLL OUT',
                               'LAND', 'LAND',
                               'HDG', 'HDG',
                               '-', '-']))


class TestAPVerticalMode(unittest.TestCase):
    
    def setUp(self):
        self._longitudinal_mode_selected_values_mapping = {
            0: '-',
            1: 'Altitude',
            2: 'Final Descent Mode',
            3: 'Flare Mode',
            4: 'Land Track Active',
            5: 'Vertical Speed Engaged',
            6: 'Unused Mode',
        }
    
    def test_can_operate(self):
        # Avoid exploding long list of combinations.
        self.assertTrue(APVerticalMode.can_operate(['Climb Active']))
        self.assertTrue(APVerticalMode.can_operate(['Longitudinal Mode Selected']))
        self.assertTrue(APVerticalMode.can_operate([
            'Climb Active',
            'Longitudinal Mode Selected',
            'ILS Glideslope Capture Active',
            'ILS Glideslope Active',
            'Flare Mode',
            'AT Active',
            'Open Climb Mode',
            'Open Descent Mode',
            'Altitude Capture Mode',
            'Altitude Mode',
            'Expedite Climb Mode',
            'Expedite Descent Mode']))
    
    def test_derive_longitudinal_mode_selected(self):
        longitudinal_mode_selected_array = np.ma.array(
            [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6])
        longitudinal_mode_selected = M(
            'Longitudinal Mode Selected',
            array=longitudinal_mode_selected_array,
            values_mapping=self._longitudinal_mode_selected_values_mapping,
        )
        node = APVerticalMode()
        node.derive(
            None, longitudinal_mode_selected, None, None, None, None, None,
            None, None, None, None, None)
        self.assertTrue(
            all(node.array ==
                ['-', '-',
                 'ALT CSTR', 'ALT CSTR',
                 'FINAL', 'FINAL',
                 'FLARE', 'FLARE',
                 'LAND', 'LAND',
                 'V/S', 'V/S',
                 '-', '-']))
    
    def test_derive_all(self):
        activated_values_mapping = {0: '-', 1: 'Activated'}
        climb_active = M(
            'Climb Active',
            array=np.ma.array([0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            values_mapping=activated_values_mapping,
        )
        ils_glideslope_capture_active = M(
            'ILS Glideslope Capture Active',
            array=np.ma.array([0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            values_mapping=activated_values_mapping,
        )
        ils_glideslope_active = M(
            'ILS Glideslope Active',
            array=np.ma.array([0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            values_mapping=activated_values_mapping,
        )
        flare_mode = M(
            'Flare Mode',
            array=np.ma.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            values_mapping={0: '-', 1: 'Engaged'},
        )
        at_active = M(
            'AT Active',
            array=np.ma.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            values_mapping=activated_values_mapping,
        )
        open_climb_mode = M(
            'Open Climb Mode',
            array=np.ma.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            values_mapping=activated_values_mapping,
        )
        open_descent_mode = M(
            'Open Descent Mode',
            array=np.ma.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            values_mapping=activated_values_mapping,
        )
        altitude_capture_mode = M(
            'Altitude Capture Mode',
            array=np.ma.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            values_mapping=activated_values_mapping,
        )
        altitude_mode = M(
            'Altitude Capture Mode',
            array=np.ma.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            values_mapping=activated_values_mapping,
        )
        expedite_climb_mode = M(
            'Expedite Climb Mode',
            array=np.ma.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            values_mapping={0: '-', 1: 'Expedite Climb Mode'},
        )
        expedite_descent_mode = M(
            'Expedite Descent Mode',
            array=np.ma.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            values_mapping={0: '-', 1: 'Expedite Descent Mode'},
        )
        longitudinal_mode_selected = M(
            'Longitudinal Mode Selected',
            array=np.ma.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 2, 2, 4, 4, 5, 5, 0, 0]),
            values_mapping=self._longitudinal_mode_selected_values_mapping,
        )
        node = APVerticalMode()
        node.derive(climb_active, longitudinal_mode_selected,
                    ils_glideslope_capture_active, ils_glideslope_active,
                    flare_mode, at_active, open_climb_mode, open_descent_mode,
                    altitude_capture_mode, altitude_mode,
                    expedite_climb_mode, expedite_descent_mode)
        self.assertTrue(
            all(node.array ==
                ['-', '-',
                 'CLB', 'CLB',
                 'GS CAPT', 'GS CAPT',
                 'GS', 'GS',
                 'FLARE', 'FLARE',
                 'DES', 'DES',
                 'OP CLB', 'OP CLB',
                 'OP DES', 'OP DES',
                 'ALT CAPT', 'ALT CAPT',
                 'ALT', 'ALT',
                 'EXPED CLB', 'EXPED CLB',
                 'EXPED DES', 'EXPED DES',
                 'ALT CSTR', 'ALT CSTR',
                 'FINAL', 'FINAL',
                 'LAND', 'LAND',
                 'V/S', 'V/S',
                 '-', '-']))


class TestAPEngaged(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = APEngaged
        self.operational_combinations = [
            ('AP (1) Engaged',),
            ('AP (2) Engaged',),
            ('AP (3) Engaged',),
            ('AP (1) Engaged', 'AP (2) Engaged'),
            ('AP (1) Engaged', 'AP (3) Engaged'),
            ('AP (2) Engaged', 'AP (3) Engaged'),
            ('AP (1) Engaged', 'AP (2) Engaged', 'AP (3) Engaged'),
        ]
    def test_single_ap(self):
        ap1 = M(array=np.ma.array(data=[0,0,1,1,0,0]),
                   values_mapping={1:'Engaged',0:'-'},
                   name='AP (1) Engaged')
        eng = APEngaged()
        eng.derive(ap1, None, None)
        expected = M(array=np.ma.array(data=[0,0,1,1,0,0]),
                   values_mapping={0: '-', 1: 'Engaged'},
                   name='AP Engaged',
                   frequency=1,
                   offset=0.1)
        ma_test.assert_array_equal(expected.array, eng.array)

    def test_dual_ap(self):
        # Two result in just "Engaged" state still
        ap1 = M(array=np.ma.array(data=[0,0,1,1,0,0]),
                   values_mapping={1:'Engaged',0:'-'},
                   name='AP (1) Engaged')
        ap2 = M(array=np.ma.array(data=[0,0,0,1,1,0]),
                   values_mapping={1:'Engaged',0:'-'},
                   name='AP (2) Engaged')
        ap3 = None
        eng = APEngaged()
        eng.derive(ap1, ap2, ap3)
        expected = M(array=np.ma.array(data=[0,0,1,1,1,0]),
                   values_mapping={0: '-', 1: 'Engaged'},
                   name='AP Engaged',
                   frequency=1,
                   offset=0.1)

        ma_test.assert_array_equal(expected.array, eng.array)

    def test_triple_ap(self):
        ap1 = M(array=np.ma.array(data=[0,0,1,1,0,0]),
                   values_mapping={1:'Engaged',0:'-'},
                   name='AP (1) Engaged',
                   frequency=1,
                   offset=0.1)
        ap2 = M(array=np.ma.array(data=[0,1,0,1,1,0]),
                   values_mapping={1:'Engaged',0:'-'},
                   name='AP (2) Engaged',
                   frequency=1,
                   offset=0.2)
        ap3 = M(array=np.ma.array(data=[0,0,1,1,1,1]),
                   values_mapping={1:'Engaged',0:'-'},
                   name='AP (3) Engaged',
                   frequency=1,
                   offset=0.4)
        eng = APEngaged()
        eng.derive(ap1, ap2, ap3)
        expected = M(array=np.ma.array(data=[0,1,1,1,1,1]),
                   values_mapping={0: '-', 1: 'Engaged'},
                   name='AP Engaged',
                   frequency=1,
                   offset=0.25)

        ma_test.assert_array_equal(expected.array, eng.array)


class TestAPUOn(unittest.TestCase):
    def test_can_operate(self):
        opts = APUOn.get_operational_combinations()
        self.assertTrue(('APU (1) On',) in opts)
        self.assertTrue(('APU (2) On',) in opts)
        self.assertTrue(('APU (1) On', 'APU (2) On') in opts)

    def test_derive(self):
        values_mapping = {0: '-', 1: 'On'}
        apu_1 = M(name='APU (1) On',
                  array=np.ma.array(data=[0, 1, 1, 1, 0, 0, 0]),
                  values_mapping=values_mapping)
        apu_2 = M(name='APU (2) On',
                  array=np.ma.array(data=[0, 0, 1, 1, 1, 0, 0]),
                  values_mapping=values_mapping)
        node = APUOn()
        node.derive(apu_1, None)
        expected = ['-'] + ['On'] * 3 + ['-'] * 3
        np.testing.assert_array_equal(node.array, expected)
        node = APUOn()
        node.derive(None, apu_2)
        expected = ['-'] * 2 + ['On'] * 3 + ['-'] * 2
        np.testing.assert_array_equal(node.array, expected)
        node = APUOn()
        node.derive(apu_1, apu_2)
        expected = ['-'] + ['On'] * 4 + ['-'] * 2
        np.testing.assert_array_equal(node.array, expected)


class TestAPURunning(unittest.TestCase):
    def test_can_operate(self):
        opts = APURunning.get_operational_combinations()
        self.assertTrue(('APU N1',) in opts)
        self.assertTrue(('APU Generator AC Voltage',) in opts)
        self.assertTrue(('APU Bleed Valve Open',) in opts)

    def test_derive_apu_n1(self):
        apu_n1 = P('APU N1', array=np.ma.array([0, 40, 80, 100, 70, 30, 0.0]))
        run = APURunning()
        run.derive(apu_n1, None, None)
        expected = ['-'] * 2 + ['Running'] * 3 + ['-'] * 2
        np.testing.assert_array_equal(run.array, expected)
    
    def test_derive_apu_voltage(self):
        apu_voltage = P('APU Generator AC Voltage',
                        array=np.ma.array([0, 115, 115, 115, 114, 0, 0]))
        run = APURunning()
        run.derive(None, apu_voltage, None)
        expected = ['-'] + ['Running'] * 4 + ['-'] * 2
        np.testing.assert_array_equal(run.array, expected)
    
    def test_derive_apu_bleed_valve_open(self):
        apu_bleed_valve_open = M('APU Bleed Valve Open',
                                 array=np.ma.array([0,1,1,0,1],
                                                   mask=[False] * 4 + [True]),
                                 values_mapping={0: '-', 1: 'Open'})
        run = APURunning()
        run.derive(None, None, apu_bleed_valve_open)
        expected = ['-'] + ['Running'] * 2 + ['-'] * 2
        np.testing.assert_array_equal(run.array, expected)


class TestAPChannelsEngaged(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = APChannelsEngaged
        self.operational_combinations = [
            ('AP (1) Engaged', 'AP (2) Engaged'),
            ('AP (1) Engaged', 'AP (3) Engaged'),
            ('AP (2) Engaged', 'AP (3) Engaged'),
            ('AP (1) Engaged', 'AP (2) Engaged', 'AP (3) Engaged'),
        ]

    def test_single_ap(self):
        # Cannot auto_land on one AP
        ap1 = M(array=np.ma.array(data=[0,0,0,0,0,0]),
                   values_mapping={1:'Engaged',0:'-'},
                   name='AP (1) Engaged')
        values_mapping = {0: '-', 1: 'Single', 2: 'Dual', 3: 'Triple'}
        eng = APChannelsEngaged()
        eng.derive(ap1, None, None)
        expected = M(array=np.ma.array(data=[0,0,0,0,0,0]),
                   values_mapping=values_mapping,
                   name='AP Channels Engaged',
                   frequency=1,
                   offset=0.1)
        ma_test.assert_array_equal(expected.array, eng.array)

    def test_dual_ap(self):
        ap1 = M(array=np.ma.array(data=[0,0,1,1,0,0]),
                   values_mapping={1:'Engaged',0:'-'},
                   name='AP (1) Engaged')
        ap2 = M(array=np.ma.array(data=[0,0,0,1,1,0]),
                   values_mapping={1:'Engaged',0:'-'},
                   name='AP (2) Engaged')
        ap3 = None
        values_mapping = {0: '-', 1: 'Single', 2: 'Dual', 3: 'Triple'}
        eng = APChannelsEngaged()
        eng.derive(ap1, ap2, ap3)
        expected = M(array=np.ma.array(data=[0, 0, 1, 2, 1, 0]),
                   values_mapping=values_mapping,
                   name='AP Channels Engaged',
                   frequency=1,
                   offset=0.1)

        ma_test.assert_array_equal(expected.array, eng.array)

    def test_triple_ap(self):
        ap1 = M(array=np.ma.array(data=[0,0,1,1,0,0]),
                   values_mapping={1:'Engaged',0:'-'},
                   name='AP (1) Engaged',
                   frequency=1,
                   offset=0.1)
        ap2 = M(array=np.ma.array(data=[0,1,0,1,1,0]),
                   values_mapping={1:'Engaged',0:'-'},
                   name='AP (2) Engaged',
                   frequency=1,
                   offset=0.2)
        ap3 = M(array=np.ma.array(data=[0,0,1,1,1,1]),
                   values_mapping={1:'Engaged',0:'-'},
                   name='AP (3) Engaged',
                   frequency=1,
                   offset=0.4)
        values_mapping = {0: '-', 1: 'Single', 2: 'Dual', 3: 'Triple'}
        eng = APChannelsEngaged()
        eng.derive(ap1, ap2, ap3)
        expected = M(array=np.ma.array(data=[0, 1, 2, 3, 2, 1]),
                   values_mapping=values_mapping,
                   name='AP Channels Engaged',
                   frequency=1,
                   offset=0.25)

        ma_test.assert_array_equal(expected.array, eng.array)


class TestConfiguration(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = Configuration

    @patch('analysis_engine.multistate_parameters.at')
    def test_can_operate(self, at):
        at.get_conf_angles.side_effect = ({}, {}, KeyError)
        self.assertTrue(self.node_class.can_operate(
            ('Flap', 'Slat', 'Model', 'Series', 'Family'),
            manufacturer=A('Manufacturer', 'Airbus'),
            model=A('Model', 'A320-201'),
            series=A('Series', 'A320-200'),
            family=A('Family', 'A320'),
        ))
        self.assertTrue(self.node_class.can_operate(
            ('Flap', 'Slat', 'Flaperon', 'Model', 'Series', 'Family'),
            manufacturer=A('Manufacturer', 'Airbus'),
            model=A('Model', 'A330-301'),
            series=A('Series', 'A330-300'),
            family=A('Family', 'A330'),
        ))
        self.assertFalse(self.node_class.can_operate(
            ('Flap', 'Slat', 'Flaperon', 'Model', 'Series', 'Family'),
            manufacturer=A('Manufacturer', 'Airbus'),
            model=A('Model', None),
            series=A('Series', None),
            family=A('Family', None),
        ))
        self.assertFalse(self.node_class.can_operate(
            ('Flap', 'Slat', 'Flaperon', 'Model', 'Series', 'Family'),
            manufacturer=A('Manufacturer', 'Airbus'),
            model=A('Model', 'A310-101'),
            series=A('Series', 'A310-100'),
            family=A('Family', 'A310'),
        ))
        self.assertFalse(self.node_class.can_operate(
            ('Flap', 'Slat', 'Flaperon', 'Model', 'Series', 'Family'),
            manufacturer=A('Manufacturer', 'Boeing'),
            model=A('Model', None),
            series=A('Series', None),
            family=A('Family', None),
        ))

    @patch('analysis_engine.multistate_parameters.at')
    def test_derive(self, at):
        at.get_conf_angles.return_value = {
            '0':    (0, 0, 0),
            '1':    (16, 0, 0),
            '1+F':  (16, 8, 5),
            '1*':   (20, 8, 10),
            '2':    (20, 14, 10),
            '2*':   (23, 14, 10),
            '3':    (23, 22, 10),
            'Full': (23, 32, 10),
        }
        _am = A('Model', 'A330-301')
        _as = A('Series', 'A330-300')
        _af = A('Family', 'A330')
        attributes = (_am, _as, _af)
        # Note: The last state is invalid...
        s = [0] * 2 + [16] * 4 + [20] * 4 + [23] * 6 + [16]
        f = [0] * 4 + [8] * 4 + [14] * 4 + [22] * 2 + [32] * 2 + [14]
        a = [0] * 4 + [5] * 2 + [10] * 10 + [10]
        z = lambda i: {x: str(x) for x in np.ma.unique(i)}
        slat = M('Slat', np.ma.array(s), values_mapping=z(s))
        flap = M('Flap', np.ma.array(f), values_mapping=z(f))
        ails = M('Flaperon', np.ma.array(a), values_mapping=z(a))
        node = self.node_class()
        node.derive(slat, flap, ails, *attributes)
        attributes = (a.value for a in attributes)
        at.get_conf_angles.assert_called_once_with(*attributes)
        self.assertEqual(node.values_mapping, AVAILABLE_CONF_STATES)
        self.assertEqual(node.units, None)
        self.assertIsInstance(node.array, MappedArray)
        values = unique_values(node.array.astype(int))
        self.assertEqual(values, {0: 2, 1: 2, 2: 2, 3: 2, 4: 2, 5: 2, 6: 2, 9: 3})


class TestDaylight(unittest.TestCase):
    def test_can_operate(self):
        expected = [('Latitude Smoothed', 'Longitude Smoothed',
                     'Start Datetime', 'HDF Duration')]
        opts = Daylight.get_operational_combinations()
        self.assertEqual(opts, expected)

    def test_daylight_aligning(self):
        lat = P('Latitude', np.ma.array([51.1789]*128), offset=0.1)
        lon = P('Longitude', np.ma.array([-1.8264]*128))
        start_dt = A('Start Datetime', datetime.datetime(2012,6,20, 20,25))
        dur = A('HDF Duration', 128)

        don = Daylight()
        don.get_derived((lat, lon, start_dt, dur))
        self.assertEqual(list(don.array), [np.ma.masked] + ['Day']*31)
        self.assertEqual(don.frequency, 0.25)
        self.assertEqual(don.offset, 0)

    def test_father_christmas(self):
        # Starting on the far side of the world, he flies all round
        # delivering parcels mostly by night (in the northern lands).
        lat=P('Latitude', np.ma.arange(60,64,1/64.0))
        lon=P('Longitude', np.ma.arange(-180,180,90/64.0))
        start_dt = A('Start Datetime', datetime.datetime(2012,12,25,01,00))
        dur = A('HDF Duration', 256)

        don = Daylight()
        don.align_frequency = 1/64.0  # Force frequency to simplify test
        don.get_derived((lat, lon, start_dt, dur))
        expected = ['Day', 'Night', 'Night', 'Night']
        np.testing.assert_array_equal(don.array, expected)  # FIX required to test as no longer superframe samples


class TestDualInputWarning(unittest.TestCase, NodeTest):
    def setUp(self):
        self.node_class = DualInputWarning
        self.operational_combinations = [
            ('Pilot Flying', 'Sidestick Angle (Capt)', 'Sidestick Angle (FO)')
        ]

    def test_derive(self):
        pilot_map = {0: '-', 1: 'Captain', 2: 'First Officer'}
        pilot_array = MappedArray([1] * 20 + [0] * 10 + [2] * 20,
                                  values_mapping=pilot_map)
        capt_array = np.ma.concatenate((15 + np.arange(20), np.zeros(30)))
        fo_array = np.ma.concatenate((np.zeros(30), 15 + np.arange(20)))
        # Dual input
        fo_array[5:10] = 15
        pilot = M('Pilot Flying', pilot_array, values_mapping=pilot_map)
        capt = P('Sidestick Angle (Capt)', capt_array)
        fo = P('Sidestick Angle (FO)', fo_array)
        node = self.node_class()
        node.derive(pilot, capt, fo)

        expected_array = MappedArray(
            np.ma.zeros(capt_array.size),
            values_mapping=self.node_class.values_mapping)
        expected_array[5:10] = 'Dual'
        np.testing.assert_array_equal(node.array, expected_array)

    def test_derive_from_hdf(self):
        (pilot, capt, fo), phase = self.get_params_from_hdf(
            os.path.join(test_data_path, 'dual_input.hdf5'),
            ['Pilot Flying', 'Sidestick Angle (Capt)', 'Sidestick Angle (FO)'])

        node = self.node_class()
        node.derive(pilot, capt, fo)

        expected_array = MappedArray(
            np.ma.zeros(pilot.array.size),
            values_mapping=self.node_class.values_mapping)
        expected_array[177:212] = 'Dual'
        np.testing.assert_array_equal(node.array, expected_array)

    def test_not_triggered_at_minimum_resolution(self):
        pilot_map = {0: '-', 1: 'Captain', 2: 'First Officer'}
        pilot_array = MappedArray([1] * 20,
                                  values_mapping=pilot_map)
        capt_array = np.ma.array([10.0]*20)
        fo_array = np.ma.array([0.0]*20)
        # Dual input
        
        # The resolution of the signals on some Airbus types is given by:
        resolution = 0.703129
        # So the sidesticks can sit with one bit offset in pitch and roll at
        # this angle (given 10% increase for safety):
        min_res = ((resolution*resolution)*2*1.1)**0.5

        fo_array[5:10] = min_res
        pilot = M('Pilot Flying', np.ma.array([1]*20), values_mapping=pilot_map)
        capt = P('Sidestick Angle (Capt)', capt_array)
        fo = P('Sidestick Angle (FO)', fo_array)
        node = self.node_class()
        node.derive(pilot, capt, fo)

        expected_array = MappedArray(
            np.ma.zeros(capt_array.size),
            values_mapping=self.node_class.values_mapping)
        expected_array[5:10] = '-'
        np.testing.assert_array_equal(node.array, expected_array)



class TestEng_1_Fire(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = Eng_1_Fire
        self.operational_combinations = [('Eng (1) Fire On Ground', 'Eng (1) Fire In Air')]

    def test_derive(self):
        fire_gnd = M(
            name='Eng (1) Fire On Ground',
            array=np.ma.array(data=[0, 0, 0, 1, 1, 1]),
            values_mapping={0: '-', 1: 'Fire'},
            frequency=1,
            offset=0.1,
        )
        fire_air = M(
            name='Eng (1) Fire On Ground',
            array=np.ma.array(data=[0, 0, 1, 1, 0, 0]),
            values_mapping={0: '-', 1: 'Fire'},
            frequency=1,
            offset=0.1,
        )
        node = self.node_class()
        node.derive(fire_gnd, fire_air)
        np.testing.assert_array_equal(node.array, [0, 0, 1, 1, 1, 1])


class TestEng_2_Fire(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = Eng_2_Fire
        self.operational_combinations = [('Eng (2) Fire On Ground', 'Eng (2) Fire In Air')]

    def test_derive(self):
        fire_gnd = M(
            name='Eng (2) Fire On Ground',
            array=np.ma.array(data=[0, 0, 0, 1, 1, 1]),
            values_mapping={0: '-', 1: 'Fire'},
            frequency=1,
            offset=0.1,
        )
        fire_air = M(
            name='Eng (2) Fire On Ground',
            array=np.ma.array(data=[0, 0, 1, 1, 0, 0]),
            values_mapping={0: '-', 1: 'Fire'},
            frequency=1,
            offset=0.1,
        )
        node = self.node_class()
        node.derive(fire_gnd, fire_air)
        np.testing.assert_array_equal(node.array, [0, 0, 1, 1, 1, 1])


class TestEng_3_Fire(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = Eng_3_Fire
        self.operational_combinations = [('Eng (3) Fire On Ground', 'Eng (3) Fire In Air')]

    def test_derive(self):
        fire_gnd = M(
            name='Eng (3) Fire On Ground',
            array=np.ma.array(data=[0, 0, 0, 1, 1, 1]),
            values_mapping={0: '-', 1: 'Fire'},
            frequency=1,
            offset=0.1,
        )
        fire_air = M(
            name='Eng (3) Fire On Ground',
            array=np.ma.array(data=[0, 0, 1, 1, 0, 0]),
            values_mapping={0: '-', 1: 'Fire'},
            frequency=1,
            offset=0.1,
        )
        node = self.node_class()
        node.derive(fire_gnd, fire_air)
        np.testing.assert_array_equal(node.array, [0, 0, 1, 1, 1, 1])


class TestEng_4_Fire(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = Eng_4_Fire
        self.operational_combinations = [('Eng (4) Fire On Ground', 'Eng (4) Fire In Air')]

    def test_derive(self):
        fire_gnd = M(
            name='Eng (4) Fire On Ground',
            array=np.ma.array(data=[0, 0, 0, 1, 1, 1]),
            values_mapping={0: '-', 1: 'Fire'},
            frequency=1,
            offset=0.1,
        )
        fire_air = M(
            name='Eng (4) Fire On Ground',
            array=np.ma.array(data=[0, 0, 1, 1, 0, 0]),
            values_mapping={0: '-', 1: 'Fire'},
            frequency=1,
            offset=0.1,
        )
        node = self.node_class()
        node.derive(fire_gnd, fire_air)
        np.testing.assert_array_equal(node.array, [0, 0, 1, 1, 1, 1])


class TestEventMarker(unittest.TestCase):

    def test_can_operate(self):
        self.assertTrue(EventMarker.can_operate(('Event Marker (1)',)))
        self.assertTrue(EventMarker.can_operate(('Event Marker (2)',)))
        self.assertTrue(EventMarker.can_operate(('Event Marker (3)',)))
        self.assertTrue(EventMarker.can_operate(('Event Marker (Capt)',)))
        self.assertTrue(EventMarker.can_operate(('Event Marker (FO)',)))
        self.assertTrue(EventMarker.can_operate(('Event Marker (1)',
                                                 'Event Marker (2)',
                                                 'Event Marker (3)')))
        self.assertTrue(EventMarker.can_operate(('Event Marker (Capt)',
                                                 'Event Marker (FO)')))

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestEng_Fire(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = Eng_Fire
        self.operational_combinations = [
            ('Eng (1) Fire',), ('Eng (2) Fire',), ('Eng (3) Fire',), ('Eng (4) Fire',),
            ('Eng (1) Fire', 'Eng (2) Fire'), ('Eng (1) Fire', 'Eng (3) Fire'),
            ('Eng (1) Fire', 'Eng (4) Fire'), ('Eng (2) Fire', 'Eng (3) Fire'),
            ('Eng (2) Fire', 'Eng (4) Fire'), ('Eng (3) Fire', 'Eng (4) Fire'),
            ('Eng (1) Fire', 'Eng (2) Fire', 'Eng (3) Fire'),
            ('Eng (1) Fire', 'Eng (2) Fire', 'Eng (4) Fire'),
            ('Eng (1) Fire', 'Eng (3) Fire', 'Eng (4) Fire'),
            ('Eng (2) Fire', 'Eng (3) Fire', 'Eng (4) Fire'),
            ('Eng (1) Fire', 'Eng (2) Fire', 'Eng (3) Fire', 'Eng (4) Fire'),
        ]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestEng_AllRunning(unittest.TestCase, NodeTest):
    def setUp(self):
        from analysis_engine.multistate_parameters import Eng_AllRunning

        self.node_class = Eng_AllRunning
        self.operational_combinations = [
            ('Eng (*) N1 Min',),
            ('Eng (*) N2 Min',), ('Eng (*) Fuel Flow Min',),
            ('Eng (*) N2 Min', 'Eng (*) Fuel Flow Min'),
        ]

    def test_derive_n2_only(self):
        n2_array = np.ma.array([0, 5, 10, 15, 11, 5, 0])
        n2 = P('Eng (*) N2 Min', array=n2_array)
        expected = [False, False, False, True, True, False, False]
        node = self.node_class()
        node.derive(None, n2, None)
        self.assertEqual(node.array.raw.tolist(), expected)

    def test_derive_ff_only(self):
        ff_array = np.ma.array([10, 20, 50, 55, 51, 15, 10])
        ff = P('Eng (*) Fuel Flow Min', array=ff_array)
        expected = [False, False, False, True, True, False, False]
        node = self.node_class()
        node.derive(None, None, ff)
        self.assertEqual(node.array.raw.tolist(), expected)

    def test_derive_n2_ff(self):
        n2_array = np.ma.array([0, 5, 11, 15, 11, 5, 0])
        n2 = P('Eng (*) N2 Min', array=n2_array)
        ff_array = np.ma.array([10, 20, 50, 55, 51, 51, 10])
        ff = P('Eng (*) Fuel Flow Min', array=ff_array)
        expected = [False, False, False, True, True, False, False]
        node = self.node_class()
        node.derive(None, n2, ff)
        self.assertEqual(node.array.raw.tolist(), expected)

    def test_derive_n1_only(self):
        # fallback to N1 if no N2 and FF
        n1_array = np.ma.array([0, 5, 10, 15, 11, 5, 0])
        n1 = P('Eng (*) N2 Min', array=n1_array)
        expected = [False, False, False, True, True, False, False]
        node = self.node_class()
        node.derive(None, n1, None)
        self.assertEqual(node.array.raw.tolist(), expected)



class TestEng_AnyRunning(unittest.TestCase, NodeTest):
    def setUp(self):
        from analysis_engine.multistate_parameters import Eng_AnyRunning

        self.node_class = Eng_AnyRunning
        self.operational_combinations = [
            ('Eng (*) N1 Max',),
            ('Eng (*) N2 Max',), ('Eng (*) Fuel Flow Max',),
            ('Eng (*) N2 Max', 'Eng (*) Fuel Flow Max'),
        ]

    def test_derive_n2_only(self):
        n2_array = np.ma.array([0, 5, 10, 15, 11, 5, 0])
        n2 = P('Eng (*) N2 Max', array=n2_array)
        expected = [False, False, False, True, True, False, False]
        node = self.node_class()
        node.derive(None, n2, None)
        self.assertEqual(node.array.raw.tolist(), expected)

    def test_derive_ff_only(self):
        ff_array = np.ma.array([10, 20, 50, 55, 51, 15, 10])
        ff = P('Eng (*) Fuel Flow Max', array=ff_array)
        expected = [False, False, False, True, True, False, False]
        node = self.node_class()
        node.derive(None, None, ff)
        self.assertEqual(node.array.raw.tolist(), expected)

    def test_derive_n2_ff(self):
        n2_array = np.ma.array([0, 5, 11, 15, 11, 5, 0])
        n2 = P('Eng (*) N2 Max', array=n2_array)
        ff_array = np.ma.array([10, 20, 50, 55, 51, 51, 10])
        ff = P('Eng (*) Fuel Flow Max', array=ff_array)
        expected = [False, False, False, True, True, False, False]
        node = self.node_class()
        node.derive(None, n2, ff)
        self.assertEqual(node.array.raw.tolist(), expected)

    def test_derive_n1_only(self):
        # fallback to N1
        n1_array = np.ma.array([0, 5, 10, 15, 11, 5, 0])
        n1 = P('Eng (*) N1 Max', array=n1_array)
        expected = [False, False, False, True, True, False, False]
        node = self.node_class()
        node.derive(n1, None, None)
        self.assertEqual(node.array.raw.tolist(), expected)


class TestEng_Oil_Press_Warning(unittest.TestCase):
    def test_can_operate(self):
        combinations = Eng_Oil_Press_Warning.get_operational_combinations()
        self.assertTrue(('Eng (1) Oil Press Low',) in combinations)
        self.assertTrue(('Eng (2) Oil Press Low',) in combinations)
        self.assertTrue(('Eng (3) Oil Press Low',) in combinations)
        self.assertTrue(('Eng (4) Oil Press Low',) in combinations)
        self.assertTrue(('Eng (1) Oil Press Low',
                         'Eng (2) Oil Press Low') in combinations)
        self.assertTrue(('Eng (1) Oil Press Low',
                         'Eng (2) Oil Press Low',
                         'Eng (3) Oil Press Low',
                         'Eng (4) Oil Press Low',) in combinations)

    def test_derive(self):
        eng_values_mapping = {0: '-', 1: 'Low Press'}
        eng_1_array = np.ma.array([1,1,0,0,1,0,1,0])
        eng_1 = M('Eng (1) Oil Press Low', array=eng_1_array,
                  values_mapping=eng_values_mapping)
        eng_2_array = np.ma.array([0,1,1,0,0,0,0,1])
        eng_2 = M('Eng (2) Oil Press Low', array=eng_1_array,
                  values_mapping=eng_values_mapping)
        node = Eng_Oil_Press_Warning()
        node.derive(eng_1, eng_2, None, None)
        node



class TestEngThrustModeRequired(unittest.TestCase):
    def test_can_operate(self):
        opts = ThrustModeSelected.get_operational_combinations()
        self.assertTrue(('Thrust Mode Selected (L)',) in opts)
        self.assertTrue(('Thrust Mode Selected (R)',) in opts)
        self.assertTrue(('Thrust Mode Selected (L)', 'Thrust Mode Selected (R)') in opts)

    def test_derive_one_param(self):
        thrust_array = np.ma.array([0, 0, 1, 0])
        thrust = M('Thrust Mode Selected (R)', array=thrust_array,
                   values_mapping=ThrustModeSelected.values_mapping)
        node = ThrustModeSelected()
        node.derive(None, thrust)
        self.assertEqual(thrust.array.raw.tolist(), thrust_array.tolist())

    def test_derive_four_params(self):
        thrust_array1 = np.ma.array([0, 0, 1, 0],
                                    mask=[False, False, True, False])
        thrust_array2 = np.ma.array([1, 0, 0, 0],
                                    mask=[True, False, False, False])
        thrust1 = M('Thrust Mode Selected (L)', array=thrust_array1,
                    values_mapping=ThrustModeSelected.values_mapping)
        thrust2 = M('Thrust Mode Selected (R)', array=thrust_array2,
                    values_mapping=ThrustModeSelected.values_mapping)
        node = ThrustModeSelected()
        node.derive(thrust1, thrust2)

        self.assertEqual(
            node.array.tolist(),
            MappedArray([1, 0, 1, 0],
                        mask=[True, False, True, False],
                        values_mapping=ThrustModeSelected.values_mapping).tolist())


class TestFlap(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = Flap

    @patch('analysis_engine.multistate_parameters.at')
    def test_can_operate(self, at):
        at.get_flap_map.side_effect = ({}, KeyError)
        self.assertTrue(self.node_class.can_operate(
            ('Flap Angle', 'Model', 'Series', 'Family'),
            model=A('Model', 'B737-333'),
            series=A('Series', 'B737-300'),
            family=A('Family', 'B737 Classic'),
        ))
        self.assertFalse(self.node_class.can_operate(
            ('Flap Angle', 'Model', 'Series', 'Family'),
            model=A('Model', None),
            series=A('Series', None),
            family=A('Family', None),
        ))
        self.assertTrue(self.node_class.can_operate(
            ('Altitude AAL',),
            frame=A('Frame', 'L382-Hercules'),
        ))

    @patch('analysis_engine.multistate_parameters.at')
    def test_derive(self, at):
        at.get_flap_map.return_value = {f: str(f) for f in (0, 1, 2, 5, 10, 15, 25, 30, 40)}
        _am = A('Model', 'B737-333')
        _as = A('Series', 'B737-300')
        _af = A('Family', 'B737 Classic')
        attributes = (_am, _as, _af)
        array = np.ma.array([0] * 5 + range(50))
        flap = P(name='Flap Angle', array=array)
        node = self.node_class()
        node.derive(flap, *attributes)
        attributes = (a.value for a in attributes)
        at.get_flap_map.assert_called_once_with(*attributes)
        self.assertEqual(node.values_mapping, at.get_flap_map.return_value)
        self.assertEqual(node.units, ut.DEGREE)
        self.assertIsInstance(node.array, MappedArray)
        values = unique_values(node.array.astype(int))
        self.assertEqual(values, {0: 6, 1: 1, 2: 1, 5: 3, 10: 5, 15: 5, 25: 10, 30: 5, 40: 19})

    @patch('analysis_engine.multistate_parameters.at')
    def test_derive__md82(self, at):
        at.get_flap_map.return_value = {f: str(f) for f in (0, 13, 20, 25, 30, 40)}
        _am = A('Model', None)
        _as = A('Series', None)
        _af = A('Family', 'DC-9')
        attributes = (_am, _as, _af)
        array = np.ma.array(range(50) + range(-5, 0) + [13.1, 1.3, 10, 10])
        flap = P(name='Flap Angle', array=array)
        for index in (1, 57, 58):
            flap.array[index] = np.ma.masked
        node = self.node_class()
        node.derive(flap, *attributes)
        attributes = (a.value for a in attributes)
        at.get_flap_map.assert_called_once_with(*attributes)
        self.assertEqual(node.values_mapping, at.get_flap_map.return_value)
        self.assertEqual(node.units, ut.DEGREE)
        self.assertIsInstance(node.array, MappedArray)
        self.assertEqual(node.array.size, 59)
        expected = np.array([0.0] + [13] * 13 + [20] * 7 + [25] * 5 + [30] * 5 + [40] * 19 + [0] * 5 + [13] + [0] * 3)
        ma_test.assert_array_equal(node.array.raw, expected)
        self.assertIsNot(node.array[1], np.ma.masked)
        self.assertIs(node.array[57], np.ma.masked)
        self.assertIs(node.array[58], np.ma.masked)

    @patch('analysis_engine.multistate_parameters.at')
    def test_derive__beechcraft(self, at):
        at.get_flap_map.return_value = {f: str(f) for f in (0, 17.5, 35)}
        _am = A('Model', '1900D')
        _as = A('Series', '1900D')
        _af = A('Family', '1900')
        attributes = (_am, _as, _af)
        array = np.ma.array((0, 5, 7.2, 17, 17.4, 17.9, 20, 30))
        flap = P(name='Flap Angle', array=array)
        node = self.node_class()
        node.derive(flap, *attributes)
        attributes = (a.value for a in attributes)
        at.get_flap_map.assert_called_once_with(*attributes)
        self.assertEqual(node.values_mapping, at.get_flap_map.return_value)
        self.assertEqual(node.units, ut.DEGREE)
        self.assertIsInstance(node.array, MappedArray)
        ma_test.assert_masked_array_equal(node.array, np.ma.array((0, 17.5, 17.5, 17.5, 17.5, 17.5, 35, 35)))

    @patch('analysis_engine.multistate_parameters.at')
    def test_derive__hercules(self, at):
        # No flap recorded; ensure it converts exactly the same...
        at.get_flap_map.return_value = {0: '0', 50: '50', 100: '100'}
        _am = A('Model', None)
        _as = A('Series', None)
        _af = A('Family', 'C-130')
        _fr = A('Frame', 'L382-Hercules')
        array = np.ma.array((0, 0, 50, 1500, 1500, 1500, 2500, 2500, 1500, 1500, 50, 50))
        alt_aal = P(name='Altitude AAL', array=array)
        node = self.node_class()
        node.derive(None, _am, _as, _af, _fr, alt_aal)
        self.assertEqual(at.get_flap_map.call_count, 0)
        self.assertEqual(node.values_mapping, at.get_flap_map.return_value)
        self.assertEqual(node.units, ut.PERCENT)
        self.assertIsInstance(node.array, MappedArray)
        ma_test.assert_masked_array_equal(node.array, np.ma.array((50, 50, 50, 0, 0, 0, 0, 0, 50, 50, 100, 100)))


class TestFlapExcludingTransition(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = FlapExcludingTransition

    @patch('analysis_engine.multistate_parameters.at')
    def test_can_operate(self, at):
        at.get_flap_map.side_effect = ({}, KeyError)
        self.assertTrue(self.node_class.can_operate(
            ('Flap Angle', 'Model', 'Series', 'Family'),
            model=A('Model', 'B737-333'),
            series=A('Series', 'B737-300'),
            family=A('Family', 'B737 Classic'),
        ))
        self.assertFalse(self.node_class.can_operate(
            ('Flap Angle', 'Model', 'Series', 'Family'),
            model=A('Model', None),
            series=A('Series', None),
            family=A('Family', None),
        ))

    @patch('analysis_engine.multistate_parameters.at')
    def test_derive(self, at):
        at.get_flap_map.return_value = {f: str(f) for f in (0, 1, 2, 5, 10, 15, 25, 30, 40)}
        _am = A('Model', 'B737-333')
        _as = A('Series', 'B737-300')
        _af = A('Family', 'B737 Classic')
        attributes = (_am, _as, _af)
        array = np.ma.array([0] * 5 + range(50))
        flap = P(name='Flap Angle', array=array)
        node = self.node_class()
        node.derive(flap, *attributes)
        attributes = (a.value for a in attributes)
        at.get_flap_map.assert_called_once_with(*attributes)
        self.assertEqual(node.values_mapping, at.get_flap_map.return_value)
        self.assertEqual(node.units, ut.DEGREE)
        self.assertIsInstance(node.array, MappedArray)
        values = unique_values(node.array.astype(int))
        self.assertEqual(values, {0: 6, 1: 1, 2: 3, 5: 5, 10: 5, 15: 10, 25: 5, 30: 10, 40: 10})


class TestFlapIncludingTransition(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = FlapIncludingTransition

    @patch('analysis_engine.multistate_parameters.at')
    def test_can_operate(self, at):
        at.get_flap_map.side_effect = ({}, KeyError)
        self.assertTrue(self.node_class.can_operate(
            ('Flap Angle', 'Model', 'Series', 'Family'),
            model=A('Model', 'B737-333'),
            series=A('Series', 'B737-300'),
            family=A('Family', 'B737 Classic'),
        ))
        self.assertFalse(self.node_class.can_operate(
            ('Flap Angle', 'Model', 'Series', 'Family'),
            model=A('Model', None),
            series=A('Series', None),
            family=A('Family', None),
        ))

    @patch('analysis_engine.multistate_parameters.at')
    def test_derive(self, at):
        at.get_flap_map.return_value = {f: str(f) for f in (0, 1, 2, 5, 10, 15, 25, 30, 40)}
        _am = A('Model', 'B737-333')
        _as = A('Series', 'B737-300')
        _af = A('Family', 'B737 Classic')
        attributes = (_am, _as, _af)
        array = np.ma.array([0] * 5 + range(50))
        flap = P(name='Flap Angle', array=array)
        node = self.node_class()
        node.derive(flap, *attributes)
        attributes = (a.value for a in attributes)
        at.get_flap_map.assert_called_once_with(*attributes)
        self.assertEqual(node.values_mapping, at.get_flap_map.return_value)
        self.assertEqual(node.units, ut.DEGREE)
        self.assertIsInstance(node.array, MappedArray)
        values = unique_values(node.array.astype(int))
        self.assertEqual(values, {0: 6, 1: 1, 2: 1, 5: 3, 10: 5, 15: 5, 25: 10, 30: 5, 40: 19})


class TestFlapLever(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = FlapLever

    def test_can_operate(self):
        self.assertTrue(self.node_class.can_operate(
            ('Flap Lever Angle', 'Model', 'Series', 'Family'),
            model=A('Model', 'B737-333'),
            series=A('Series', 'B737-300'),
            family=A('Family', 'B737 Classic')),
        )

    @unittest.skip('Test not implemented.')
    def test_derive(self):
        pass


class TestFlapLeverSynthetic(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = FlapLeverSynthetic

    def test_can_operate(self):
        with patch('analysis_engine.multistate_parameters.at') as at:
            at.get_conf_angles.side_effect = KeyError
            at.get_lever_angles.return_value = {}
            # Test we can operate for something in the lever mapping.
            self.assertTrue(self.node_class.can_operate(
                ('Flap', 'Slat', 'Model', 'Series', 'Family'),
                model=A('Model', 'CRJ900 (CL-600-2D24)'),
                series=A('Series', 'CRJ900'),
                family=A('Family', 'CL-600'),
            ))
            # Test we can operate for the above even though slat missing.
            self.assertTrue(self.node_class.can_operate(
                ('Flap', 'Model', 'Series', 'Family'),
                model=A('Model', 'CRJ900 (CL-600-2D24)'),
                series=A('Series', 'CRJ900'),
                family=A('Family', 'CL-600'),
            ))
        # Test we can operate for something not in the lever mapping.
        self.assertTrue(self.node_class.can_operate(
            ('Flap', 'Model', 'Series', 'Family'),
            model=A('Model', 'B737-333'),
            series=A('Series', 'B737-300'),
            family=A('Family', 'B737 Classic'),
        ))
        # Test we can operate even though the above *has* a slat.
        self.assertTrue(self.node_class.can_operate(
            ('Flap', 'Slat', 'Model', 'Series', 'Family'),
            model=A('Model', 'B737-333'),
            series=A('Series', 'B737-300'),
            family=A('Family', 'B737 Classic'),
        ))
        # Test we can operate with Flaperons
        self.assertTrue(self.node_class.can_operate(
            ('Flap', 'Slat', 'Flaperon', 'Model', 'Series', 'Family'),
            model=A('Model', None),
            series=A('Series', None),
            family=A('Family', 'A330'),
        ))
        
        with patch('analysis_engine.multistate_parameters.at') as at:
            at.get_conf_angles.side_effect = KeyError
            # Requires Slat.
            at.get_lever_angles.return_value = {
                'Lever 0': (0, 0, None),
                'Lever 1': (20, 9, None),
                'Lever 2': (20, 20, None),
                'Lever 3': (20, 40, None),
            }
            self.assertFalse(self.node_class.can_operate(
                ('Flap', 'Model', 'Series', 'Family'),
                model=A('Model', 'CRJ900 (CL-600-2D24)'),
                series=A('Series', 'CRJ900'),
                family=A('Family', 'CL-600'),
            ))
            self.assertTrue(self.node_class.can_operate(
                ('Flap', 'Slat', 'Model', 'Series', 'Family'),
                model=A('Model', 'CRJ900 (CL-600-2D24)'),
                series=A('Series', 'CRJ900'),
                family=A('Family', 'CL-600'),
            ))
            # Requires Flaperon.
            at.get_lever_angles.return_value = {
                'Lever 0': (None, 0, 1),
                'Lever 1': (None, 9, 2),
                'Lever 2': (None, 20, 3),
                'Lever 3': (None, 40, 4),
            }
            self.assertFalse(self.node_class.can_operate(
                ('Flap', 'Model', 'Series', 'Family'),
                model=A('Model', 'CRJ900 (CL-600-2D24)'),
                series=A('Series', 'CRJ900'),
                family=A('Family', 'CL-600'),
            ))
            self.assertTrue(self.node_class.can_operate(
                ('Flap', 'Flaperon', 'Model', 'Series', 'Family'),
                model=A('Model', 'CRJ900 (CL-600-2D24)'),
                series=A('Series', 'CRJ900'),
                family=A('Family', 'CL-600'),
            ))
            # Requires Slat and Flaperon.
            at.get_lever_angles.return_value = {
                'Lever 0': (0, 0, 1),
                'Lever 1': (20, 9, 2),
                'Lever 2': (20, 20, 3),
                'Lever 3': (20, 40, 4),
            }
            self.assertFalse(self.node_class.can_operate(
                ('Flap', 'Model', 'Series', 'Family'),
                model=A('Model', 'CRJ900 (CL-600-2D24)'),
                series=A('Series', 'CRJ900'),
                family=A('Family', 'CL-600'),
            ))
            self.assertFalse(self.node_class.can_operate(
                ('Flap', 'Slat', 'Model', 'Series', 'Family'),
                model=A('Model', 'CRJ900 (CL-600-2D24)'),
                series=A('Series', 'CRJ900'),
                family=A('Family', 'CL-600'),
            ))
            self.assertFalse(self.node_class.can_operate(
                ('Flap', 'Flaperon', 'Model', 'Series', 'Family'),
                model=A('Model', 'CRJ900 (CL-600-2D24)'),
                series=A('Series', 'CRJ900'),
                family=A('Family', 'CL-600'),
            ))
            self.assertTrue(self.node_class.can_operate(
                ('Flap', 'Slat', 'Flaperon', 'Model', 'Series', 'Family'),
                model=A('Model', 'CRJ900 (CL-600-2D24)'),
                series=A('Series', 'CRJ900'),
                family=A('Family', 'CL-600'),
            ))

    @patch('analysis_engine.multistate_parameters.at')
    def test_derive__crj900(self, at):
        at.get_conf_angles.side_effect = KeyError
        at.get_lever_map.return_value = {f: str(f) for f in (0, 1, 8, 20, 30, 45)}
        at.get_lever_angles.return_value = {
            '0':  (0, 0, None),
            '1':  (20, 0, None),
            '8':  (20, 8, None),
            '20': (20, 20, None),
            '30': (25, 30, None),
            '45': (25, 45, None),
        }
        # Prepare our generated flap and slat arrays:
        flap_array = [0.0, 0, 8, 8, 0, 0, 0, 8, 20, 30, 45, 8, 0, 0, 0]
        slat_array = [0.0, 0, 20, 20, 20, 0, 0, 20, 20, 25, 25, 20, 20, 0, 0]
        flap_array = MappedArray(np.repeat(flap_array, 10), 
                values_mapping={f: str(f) for f in (0, 8, 20, 30, 45)})
        slat_array = MappedArray(np.repeat(slat_array, 10),
                values_mapping={s: str(s) for s in (0, 20, 25)})

        ### Add some noise to make our flap and slat angles more realistic:
        ##flap_array += np.ma.sin(range(len(flap_array))) * 0.1
        ##slat_array -= np.ma.sin(range(len(slat_array))) * 0.1

        # Derive the synthetic flap lever:
        flap = M('Flap', flap_array)
        slat = M('Slat', slat_array)
        flaperon = None
        model = A('Model', 'CRJ900 (CL-600-2D24)')
        series = A('Series', 'CRJ900')
        family = A('Family', 'CL-600')
        node = self.node_class()
        node.derive(flap, slat, flaperon, model, series, family)

        # Check against an expected array of lever detents:
        expected = [0, 0, 8, 8, 1, 0, 0, 8, 20, 30, 45, 8, 1, 0, 0]
        mapping = {x: str(x) for x in sorted(set(expected))}
        array = MappedArray(np.repeat(expected, 10), values_mapping=mapping)
        np.testing.assert_array_equal(node.array, array)

    def test_derive__b737ng(self):
        # Prepare our generated flap array:
        flap_array = [0.0, 0, 5, 2, 1, 0, 0, 10, 15, 25, 30, 40, 0, 0, 0]
        flap_array = MappedArray(np.repeat(flap_array, 10),
                values_mapping={f: str(f) for f in (0, 1, 2, 5, 10, 15, 25, 30, 40)})

        ### Add some noise to make our flap angles more realistic:
        ##flap_array += np.ma.sin(range(len(flap_array))) * 0.05

        # Derive the synthetic flap lever:
        flap = M('Flap', flap_array)
        slat = None
        flaperon = None
        model = A('Model', 'B737-333')
        series = A('Series', 'B737-300')
        family = A('Family', 'B737 Classic')
        node = self.node_class()
        node.derive(flap, slat, flaperon, model, series, family)

        # Check against an expected array of lever detents:
        expected = [0, 0, 5, 2, 1, 0, 0, 10, 15, 25, 30, 40, 0, 0, 0]
        mapping = {x: str(x) for x in sorted(set(expected))}
        array = MappedArray(np.repeat(expected, 10), values_mapping=mapping)
        np.testing.assert_array_equal(node.array, array)

    def test_derive__a330(self):
        # A330 uses Configuration conditions.
        # Prepare our generated flap and slat arrays:
        #                 -  - 1+F   -  -  1   1*  2   2*  3  Full -
        slat_array =     [0, 0, 16, 16, 0, 16, 20, 20, 23, 23, 23, 0]
        flap_array =     [0, 0,  8,  8, 0,  0,  8, 14, 14, 22, 32, 0]
        flaperon_array = [0, 0,  5,  0, 0,  0, 10, 10, 10, 10, 10, 0]
        expected = ['Lever 0', 'Lever 0', 'Lever 1', 'Lever 0', 'Lever 0',
                    'Lever 1', 'Lever 2', 'Lever 2', 'Lever 3', 'Lever 3',
                    'Lever Full', 'Lever 0']
        repeat = 1
        flap_array = MappedArray(np.repeat(flap_array, repeat), 
                values_mapping={f: str(f) for f in (0, 8, 14, 22, 32)})
        slat_array = MappedArray(np.repeat(slat_array, repeat),
                values_mapping={s: str(s) for s in (0, 16, 20, 23)})
        flaperon_array = MappedArray(np.repeat(flaperon_array, repeat),
                values_mapping={a: str(a) for a in (0, 5, 10)})

        # Derive the synthetic flap lever:
        flap = M('Flap', flap_array)
        slat = M('Slat', slat_array)
        flaperon = M('Flaperon', flaperon_array)
        model = A('Model', None)
        series = A('Series', None)
        family = A('Family', 'A330')
        node = self.node_class()
        node.derive(flap, slat, flaperon, model, series, family)

        mapping = {x: str(x) for x in sorted(set(expected))}
        self.assertTrue(np.all(node.array == np.repeat(expected, repeat)))


class TestFlaperon(unittest.TestCase):
    def test_can_operate(self):
        self.assertTrue(Flaperon.can_operate(
            ('Aileron (L)', 'Aileron (R)', 'Model', 'Series', 'Family'),
            model=Attribute('Model', 'A330-222'),
            series=Attribute('Series', 'A330-200'),
            family=Attribute('Family', 'A330')))

    def test_derive(self):
        al = load(os.path.join(test_data_path, 'aileron_left.nod'))
        ar = load(os.path.join(test_data_path, 'aileron_right.nod'))
        model = A('Model', 'A330-222')
        series = A('Series', 'A330-200')
        family = A('Family', 'A330')
        flaperon = Flaperon()
        flaperon.derive(al, ar, model, series, family)
        # ensure values are grouped into aileron settings accordingly
        # flaperon is now step at movement start
        self.assertEqual(unique_values(flaperon.array.astype(int)),{
            0: 22056,
            5: 282,
            10: 1148})


class TestFuelQtyLow(unittest.TestCase):
    def test_can_operate(self):
        opts = FuelQty_Low.get_operational_combinations()
        self.assertIn(('Fuel Qty Low',), opts)
        self.assertIn(('Fuel Qty (L) Low',), opts)
        self.assertIn(('Fuel Qty (R) Low',), opts)
        self.assertIn(('Fuel Qty (L) Low', 'Fuel Qty (R) Low'), opts)

    def test_derive_fuel_qty_low_warning(self):
        low = M(array=np.ma.array([0,0,0,1,1,0]), values_mapping={1: 'Warning'})
        warn = FuelQty_Low()
        warn.derive(low, None, None)
        self.assertEqual(warn.array.sum(), 2)

    def test_derive_fuel_qty_low_warning_two_params(self):
        one = M(array=np.ma.array([0,0,0,1,1,0]), values_mapping={1: 'Warning'})
        two = M(array=np.ma.array([0,0,1,1,0,0]), values_mapping={1: 'Warning'})
        warn = FuelQty_Low()
        warn.derive(None, one, two)
        self.assertEqual(warn.array.sum(), 3)


class TestGearDown(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = GearDown
        self.operational_combinations = [
            ('Gear (L) Down',),
            ('Gear (R) Down',),
            ('Gear (L) Down', 'Gear (R) Down'),
            ('Gear (L) Down', 'Gear (N) Down', 'Gear (R) Down'),
            ('Gear Up', 'Gear In Transit'),
            ('Gear Down Selected',),
        ]

    def test_derive_from_select_down(self):
        sel_down = M(array=np.ma.array([1,0,0,1,1]), values_mapping={
            0: 'Up',
            1: 'Down',
        })
        down = GearDown()
        down.derive(None, None, None, None, None, sel_down)
        self.assertEqual(list(down.array),
                         ['Down', 'Up', 'Up', 'Down', 'Down'])

    def test_derive_from_up_intransit(self):
        up = M(array=np.ma.array([0,0,0,0,1,1,0,0,0,0]), values_mapping={
            1: 'Up',
            0: 'Down',
        })
        in_transit = M(array=np.ma.array([0,0,0,1,0,0,1,1,0,0]), values_mapping={
            1: 'In Transit',
            0: '-',
        })
        node = self.node_class()
        node.derive(None, None, None, up, in_transit, None)
        self.assertEqual(list(node.array),
                 ['Down', 'Down', 'Down', 'Up', 'Up', 'Up', 'Up', 'Up', 'Down', 'Down'])

class TestGearUp(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = GearUp
        self.operational_combinations = [
            ('Gear (L) Up',),
            ('Gear (R) Up',),
            ('Gear (L) Up', 'Gear (R) Up'),
            ('Gear (L) Up', 'Gear (N) Up', 'Gear (R) Up'),
        ]


class TestGearInTransit(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = GearInTransit
        self.operational_combinations = [
            ('Gear (L) In Transit',),
            ('Gear (R) In Transit',),
            ('Gear (L) In Transit', 'Gear (R) In Transit'),
            ('Gear (L) In Transit', 'Gear (N) In Transit', 'Gear (R) In Transit'),
        ]



class TestGearUpSelected(unittest.TestCase):
    def test_can_operate(self):
        opts = GearUpSelected.get_operational_combinations()
        self.assertEqual(opts, [('Gear Down Selected',)])

    def test_gear_up_selected_from_recorded_down(self):
        dn_sel = M(
            array=np.ma.array(data=[1, 1, 0, 0, 1, 1]),
            values_mapping={0: 'Down', 1: 'Up'},
            name='Gear Down Selected',
            frequency=1,
            offset=0.1
        )
        up_sel = GearUpSelected()
        up_sel.derive(dn_sel)
        np.testing.assert_array_equal(up_sel.array, [0, 0, 1, 1, 0, 0])


class TestGearOnGround(unittest.TestCase):
    def test_can_operate(self):
        opts = GearOnGround.get_operational_combinations()
        self.assertEqual(opts, [
            ('Gear (L) On Ground',),
            ('Gear (R) On Ground',),
            ('Gear (L) On Ground', 'Gear (R) On Ground'),
            ])

    def test_gear_on_ground_basic(self):
        p_left = M(array=np.ma.array(data=[0,0,1,1]),
                   values_mapping={0:'Air',1:'Ground'},
                   name='Gear (L) On Ground',
                   frequency=1,
                   offset=0.1)
        p_right = M(array=np.ma.array(data=[0,1,1,1]),
                    values_mapping={0:'Air',1:'Ground'},
                    name='Gear (R) On Ground',
                    frequency=1,
                    offset=0.6)
        wow=GearOnGround()
        wow.derive(p_left, p_right)
        np.testing.assert_array_equal(wow.array, [0,0,0,1,1,1,1,1])
        self.assertEqual(wow.frequency, 2.0)
        self.assertAlmostEqual(wow.offset, 0.1)

    def test_gear_on_ground_common_word(self):
        p_left = M(array=np.ma.array(data=[0,0,1,1]),
                   values_mapping={0:'Air',1:'Ground'},
                   name='Gear (L) On Ground',
                   frequency=1,
                   offset=0.1)
        p_right = M(array=np.ma.array(data=[0,1,1,1]),
                    values_mapping={0:'Air',1:'Ground'},
                    name='Gear (R) On Ground',
                    frequency=1,
                    offset=0.1)
        wow=GearOnGround()
        wow.derive(p_left, p_right)
        np.testing.assert_array_equal(wow.array, [0,1,1,1])
        self.assertEqual(wow.frequency, 1.0)
        self.assertAlmostEqual(wow.offset, 0.1)

    def test_gear_on_ground_left_only(self):
        p_left = M(array=np.ma.array(data=[0,0,1,1]),
                   values_mapping={0:'Air',1:'Ground'},
                   name='Gear (L) On Ground',
                   frequency=1,
                   offset=0.1)
        wow=GearOnGround()
        wow.derive(p_left, None)
        np.testing.assert_array_equal(wow.array, [0,0,1,1])
        self.assertEqual(wow.frequency, 1.0)
        self.assertAlmostEqual(wow.offset, 0.1)

    def test_gear_on_ground_right_only(self):
        p_right = M(array=np.ma.array(data=[0,0,0,1]),
                    values_mapping={0:'Air',1:'Ground'},
                    name='Gear (R) On Ground',
                    frequency=1,
                    offset=0.7)
        wow=GearOnGround()
        wow.derive(None, p_right)
        np.testing.assert_array_equal(wow.array, [0,0,0,1])
        self.assertEqual(wow.frequency, 1.0)
        self.assertAlmostEqual(wow.offset, 0.7)


class TestGearDownSelected(unittest.TestCase):
    def test_can_operate(self):
        opts = GearDownSelected.get_operational_combinations()
        self.assertEqual(opts, [('Gear Down',),
                                ('Gear Down', 'Gear (*) Red Warning')])

    def test_gear_dn_selected_basic(self):
        gdn = M(
            array=np.ma.array(data=[1, 1, 1, 0, 0, 0]),
            values_mapping={1: 'Down', 0: 'Up'},
            name='Gear Down',
            frequency=1,
            offset=0.1
        )
        dn_sel = GearDownSelected()
        dn_sel.derive(gdn, None)
        np.testing.assert_array_equal(dn_sel.array, [1, 1, 1, 0, 0, 0])

    def test_gear_dn_selected_with_warnings(self):
        gdn = M(
            array=np.ma.array(data=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]),
            values_mapping={1: 'Down', 0: 'Up'},
            name='Gear Down',
            frequency=1,
            offset=0.1
        )
        redl = M(
            array=np.ma.array(data=[0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0]),
            values_mapping={0: '-', 1: 'Warning'},
            name='Gear (L) Red Warning',
            frequency=1,
            offset=0.6
        )
        redr = redl
        redn = M(
            array=np.ma.array(data=[0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0]),
            values_mapping={0: '-', 1: 'Warning'},
            name='Gear (N) Red Warning',
            frequency=1,
            offset=0.6
        )
        # fully airborne sample
        airs = S(items=[Section('Airborne', slice(None), None, None)])
        gear_warn = Gear_RedWarning()
        gear_warn.derive(redl, redn, redr, airs)
        # gear selected
        dn_sel = GearDownSelected()
        dn_sel.derive(gdn, gear_warn)
        np.testing.assert_array_equal(
            dn_sel.array.raw,
            [1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1])


class TestGear_RedWarning(unittest.TestCase):

    def test_can_operate(self):
        opts = Gear_RedWarning.get_operational_combinations()
        self.assertEqual(len(opts), 7)
        self.assertIn(('Gear (L) Red Warning', 'Airborne'), opts)
        self.assertIn(('Gear (L) Red Warning',
                       'Gear (N) Red Warning',
                       'Gear (R) Red Warning',
                       'Airborne'), opts)

    def test_derive(self):
        gear_warn_l = M('Gear (L) Red Warning',
                        np.ma.array([0,0,0,1,0,0,0,0,0,1,0,0]),
                        values_mapping={1:'Warning', 0:'-'})
        gear_warn_l.array[0] = np.ma.masked
        gear_warn_n = M('Gear (N) Red Warning',
                        np.ma.array([0,1,0,0,1,0,0,0,1,0,0,0]),
                        values_mapping={1:'Warning', 0:'-'})
        gear_warn_r = M('Gear (R) Red Warning',
                        np.ma.array([0,0,0,0,0,1,0,1,0,0,0,0]),
                        values_mapping={1:'Warning', 0:'-'})
        airs = S(items=[Section('Airborne', slice(2, 11), 2, 10)])
        gear_warn = Gear_RedWarning()
        gear_warn.derive(gear_warn_l, gear_warn_n, gear_warn_r, airs)
        self.assertEqual(list(gear_warn.array),
                         ['-', '-', '-', 'Warning', 'Warning', 'Warning',
                          '-', 'Warning', 'Warning', 'Warning', '-', '-'])

class TestKeyVHFCapt(unittest.TestCase):

    def test_can_operate(self):
        self.assertEqual(KeyVHFCapt.get_operational_combinations(),
                         [('Key VHF (1) (Capt)',),
                          ('Key VHF (2) (Capt)',),
                          ('Key VHF (3) (Capt)',),
                          ('Key VHF (1) (Capt)', 'Key VHF (2) (Capt)'),
                          ('Key VHF (1) (Capt)', 'Key VHF (3) (Capt)'),
                          ('Key VHF (2) (Capt)', 'Key VHF (3) (Capt)'),
                          ('Key VHF (1) (Capt)', 'Key VHF (2) (Capt)', 'Key VHF (3) (Capt)')])

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        pass


class TestKeyVHFFO(unittest.TestCase):

    def test_can_operate(self):
        self.assertEqual(KeyVHFFO.get_operational_combinations(),
                         [('Key VHF (1) (FO)',),
                          ('Key VHF (2) (FO)',),
                          ('Key VHF (3) (FO)',),
                          ('Key VHF (1) (FO)', 'Key VHF (2) (FO)'),
                          ('Key VHF (1) (FO)', 'Key VHF (3) (FO)'),
                          ('Key VHF (2) (FO)', 'Key VHF (3) (FO)'),
                          ('Key VHF (1) (FO)', 'Key VHF (2) (FO)', 'Key VHF (3) (FO)')])

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        pass


class TestMasterWarning(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = MasterWarning
        self.operational_combinations = [
            ('Master Warning (Capt)',),
            ('Master Warning (FO)',),
            ('Master Warning (Capt)', 'Master Warning (FO)'),
        ]

    def test_derive(self):
        warn_capt = M(
            name='Master Warning (Capt)',
            array=np.ma.array(data=[0, 0, 0, 1, 1, 1]),
            values_mapping={0: '-', 1: 'Warning'},
            frequency=1,
            offset=0.1,
        )
        warn_fo = M(
            name='Master Warning (FO)',
            array=np.ma.array(data=[0, 0, 1, 1, 0, 0]),
            values_mapping={0: '-', 1: 'Warning'},
            frequency=1,
            offset=0.1,
        )
        node = self.node_class()
        node.derive(warn_capt, warn_fo)
        np.testing.assert_array_equal(node.array, [0, 0, 1, 1, 1, 1])


class TestPackValvesOpen(unittest.TestCase):
    def test_can_operate(self):
        opts = PackValvesOpen.get_operational_combinations()
        self.assertEqual(opts, [
            ('ECS Pack (1) On', 'ECS Pack (2) On'),
            ('ECS Pack (1) On', 'ECS Pack (1) High Flow', 'ECS Pack (2) On'),
            ('ECS Pack (1) On', 'ECS Pack (2) On', 'ECS Pack (2) High Flow'),
            ('ECS Pack (1) On', 'ECS Pack (1) High Flow', 'ECS Pack (2) On', 'ECS Pack (2) High Flow')])

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestPilotFlying(unittest.TestCase, NodeTest):
    def setUp(self):
        self.node_class = PilotFlying
        self.operational_combinations = [
            ('Sidestick Angle (Capt)', 'Sidestick Angle (FO)'),
        ]

    def test_derive(self):
        stick_capt_array = np.ma.concatenate((np.ma.zeros(100),
                                              np.ma.zeros(100) + 20))
        stick_fo_array = np.ma.concatenate((np.ma.zeros(100) + 20,
                                            np.ma.zeros(100)))
        stick_capt = P('Sidestick Angle (Capt)', array=stick_capt_array)
        stick_fo = P('Sidestick Angle (FO)', array=stick_fo_array)
        node = self.node_class()
        node.derive(stick_capt, stick_fo)
        expected_array = MappedArray([2.] * 100 + [1.] * 100)
        expected = M('Pilot Flying', expected_array,
                     values_mapping=PilotFlying.values_mapping)
        print node.array
        print expected.array, expected.values_mapping
        np.testing.assert_array_equal(node.array, expected.array)


class TestPitchAlternateLaw(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = PitchAlternateLaw
        self.operational_combinations = [
            ('Pitch Alternate Law (1)',),
            ('Pitch Alternate Law (2)',),
            ('Pitch Alternate Law (1)', 'Pitch Alternate Law (2)'),
        ]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')

    def test_derive_basic(self):
        alt_law1 = M(
            name='Pitch Alternate Law (1)',
            array=np.ma.array([0, 0, 0, 1, 1, 1]),
            values_mapping={0: '-', 1: 'Engaged'},
            frequency=1,
            offset=0.1,
        )
        alt_law2 = M(
            name='Pitch Alternate Law (2)',
            array=np.ma.array([0, 0, 1, 1, 0, 0]),
            values_mapping={0: '-', 1: 'Engaged'},
            frequency=1,
            offset=0.1,
        )
        node = self.node_class()
        node.derive(alt_law1, alt_law2)
        np.testing.assert_array_equal(node.array, [0, 0, 1, 1, 1, 1])


class TestSlat(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = Slat

    @patch('analysis_engine.multistate_parameters.at')
    def test_can_operate(self, at):
        at.get_slat_map.side_effect = ({}, KeyError)
        self.assertTrue(self.node_class.can_operate(
            ('Slat Angle', 'Model', 'Series', 'Family'),
            model=A('Model', 'CRJ900 (CL-600-2D24)'),
            series=A('Series', 'CRJ900'),
            family=A('Family', 'CL-600'),
        ))
        self.assertFalse(self.node_class.can_operate(
            ('Slat Angle', 'Model', 'Series', 'Family'),
            model=A('Model', 'B737-333'),
            series=A('Series', 'B737-300'),
            family=A('Family', 'B737 Classic'),
        ))

    @patch('analysis_engine.multistate_parameters.at')
    def test_derive(self, at):
        at.get_slat_map.return_value = {s: str(s) for s in (0, 16, 25)}
        _am = A('Model', None)
        _as = A('Series', 'A300B4')
        _af = A('Family', 'A300')
        attributes = (_am, _as, _af)
        array = np.ma.array([0] * 5 + range(50))
        slat = P(name='Slat Angle', array=array)
        node = self.node_class()
        node.derive(slat, *attributes)
        attributes = (a.value for a in attributes)
        at.get_slat_map.assert_called_once_with(*attributes)
        self.assertEqual(node.values_mapping, at.get_slat_map.return_value)
        self.assertEqual(node.units, ut.DEGREE)
        self.assertIsInstance(node.array, MappedArray)
        values = unique_values(node.array.astype(int))
        self.assertEqual(values, {0: 6, 16: 16, 25: 33})


class TestSlatExcludingTransition(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = SlatExcludingTransition

    @patch('analysis_engine.multistate_parameters.at')
    def test_can_operate(self, at):
        at.get_slat_map.side_effect = ({}, KeyError)
        self.assertTrue(self.node_class.can_operate(
            ('Slat Angle', 'Model', 'Series', 'Family'),
            model=A('Model', 'CRJ900 (CL-600-2D24)'),
            series=A('Series', 'CRJ900'),
            family=A('Family', 'CL-600'),
        ))
        self.assertFalse(self.node_class.can_operate(
            ('Slat Angle', 'Model', 'Series', 'Family'),
            model=A('Model', 'B737-333'),
            series=A('Series', 'B737-300'),
            family=A('Family', 'B737 Classic'),
        ))

    @patch('analysis_engine.multistate_parameters.at')
    def test_derive(self, at):
        at.get_slat_map.return_value = {s: str(s) for s in (0, 16, 25)}
        _am = A('Model', None)
        _as = A('Series', 'A300B4')
        _af = A('Family', 'A300')
        attributes = (_am, _as, _af)
        array = np.ma.array([0] * 5 + range(50))
        slat = P(name='Slat Angle', array=array)
        node = self.node_class()
        node.derive(slat, *attributes)
        attributes = (a.value for a in attributes)
        at.get_slat_map.assert_called_once_with(*attributes)
        self.assertEqual(node.values_mapping, at.get_slat_map.return_value)
        self.assertEqual(node.units, ut.DEGREE)
        self.assertIsInstance(node.array, MappedArray)
        values = unique_values(node.array.astype(int))
        self.assertEqual(values, {0: 21, 16: 9, 25: 25})


class TestSlatIncludingTransition(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = SlatIncludingTransition

    @patch('analysis_engine.multistate_parameters.at')
    def test_can_operate(self, at):
        at.get_slat_map.side_effect = ({}, KeyError)
        self.assertTrue(self.node_class.can_operate(
            ('Slat Angle', 'Model', 'Series', 'Family'),
            model=A('Model', 'CRJ900 (CL-600-2D24)'),
            series=A('Series', 'CRJ900'),
            family=A('Family', 'CL-600'),
        ))
        self.assertFalse(self.node_class.can_operate(
            ('Slat Angle', 'Model', 'Series', 'Family'),
            model=A('Model', 'B737-333'),
            series=A('Series', 'B737-300'),
            family=A('Family', 'B737 Classic'),
        ))

    @patch('analysis_engine.multistate_parameters.at')
    def test_derive(self, at):
        at.get_slat_map.return_value = {s: str(s) for s in (0, 16, 25)}
        _am = A('Model', None)
        _as = A('Series', 'A300B4')
        _af = A('Family', 'A300')
        attributes = (_am, _as, _af)
        array = np.ma.array([0] * 5 + range(50))
        slat = P(name='Slat Angle', array=array)
        node = self.node_class()
        node.derive(slat, *attributes)
        attributes = (a.value for a in attributes)
        at.get_slat_map.assert_called_once_with(*attributes)
        self.assertEqual(node.values_mapping, at.get_slat_map.return_value)
        self.assertEqual(node.units, ut.DEGREE)
        self.assertIsInstance(node.array, MappedArray)
        values = unique_values(node.array.astype(int))
        self.assertEqual(values, {0: 6, 16: 16, 25: 33})


class TestSpeedbrakeDeployed(unittest.TestCase):

    def test_can_operate(self):
        node = self.node_class()
        operational_combinations = node.get_operational_combinations()
        self.assertTrue(('Speedbrake (L) Deployed', 'Speedbrake (R) Deployed') in operational_combinations)
        self.assertTrue(('Speedbrake (L) Outboard Deployed',
                          'Speedbrake (R) Outboard Deployed',
                          'Speedbrake (L) Inboard Deployed',
                          'Speedbrake (R) Inboard Deployed') in operational_combinations)

    def setUp(self):
        deployed_l_array = [ 0,  0,  0,  0,  0,  1,  1,  0,  0,  0]
        deployed_r_array = [ 0,  0,  0,  0,  1,  1,  1,  0,  0,  0]

        self.deployed_l = M(name='Speedbrake (L) Deployed', array=np.ma.array(deployed_l_array), values_mapping={1:'Deployed'})
        self.deployed_r = M(name='Speedbrake (R) Deployed', array=np.ma.array(deployed_r_array), values_mapping={1:'Deployed'})
        self.node_class = SpeedbrakeDeployed

    def test_derive(self):
        result = [ 0,  0,  0,  0,  0,  1,  1,  0,  0,  0]
        node = self.node_class()
        node.derive(self.deployed_l,
                    self.deployed_r,
                    None,
                    None,
                    None,
                    None,)
        np.testing.assert_equal(node.array.data, result)

    def test_derive_masked_value(self):
        self.deployed_l.array.mask = [ 0,  0,  0,  1,  0,  1,  0,  0,  1,  0]
        self.deployed_r.array.mask = [ 0,  0,  0,  0,  0,  1,  0,  0,  1,  0]

        result_array = [ 0,  0,  0,  0,  0,  0,  1,  0,  0,  0]
        result_mask =  [ 0,  0,  0,  0,  0,  1,  0,  0,  1,  0]

        node = self.node_class()
        node.derive(self.deployed_l,
                    self.deployed_r,
                    None,
                    None,
                    None,
                    None,)
        np.testing.assert_equal(node.array.data, result_array)
        np.testing.assert_equal(node.array.mask, result_mask)


class TestSpeedbrakeSelected(unittest.TestCase):

    def test_can_operate(self):
        opts = SpeedbrakeSelected.get_operational_combinations()
        self.assertTrue(('Speedbrake Deployed',) in opts)
        self.assertTrue(('Speedbrake', 'Family') in opts)
        self.assertTrue(('Speedbrake Handle', 'Family') in opts)
        self.assertTrue(('Speedbrake Handle', 'Speedbrake', 'Family') in opts)

    def test_derive(self):
        # test with deployed
        spd_sel = SpeedbrakeSelected()
        spd_sel.derive(
            deployed=M(array=np.ma.array(
                [0, 0, 0, 1, 1, 0]), values_mapping={1:'Deployed'}),
            armed=M(array=np.ma.array(
                [0, 0, 1, 1, 0, 0]), values_mapping={1:'Armed'})
        )
        self.assertEqual(list(spd_sel.array),
            ['Stowed', 'Stowed', 'Armed/Cmd Dn', 'Deployed/Cmd Up', 'Deployed/Cmd Up', 'Stowed'])

    def test_derive_from_handle(self):
        handle_array = np.ma.concatenate([np.ma.arange(0, 2, 0.1),
                                          np.ma.arange(2, 0, -0.1)])
        spd_sel = SpeedbrakeSelected()
        array = spd_sel.derive_from_handle(handle_array)
        self.assertEqual(list(array), # MappedArray .tolist() does not output states.
                         ['Stowed']*10+['Deployed/Cmd Up']*20+['Stowed']*10)

        handle_array = np.ma.concatenate([np.ma.arange(0, 2, 0.1),
                                          np.ma.array([13]*10),
                                          np.ma.arange(0.99, 0, -0.1)])
        spd_sel = SpeedbrakeSelected()
        array = spd_sel.derive_from_handle(handle_array, deployed=5, armed=1)
        self.assertEqual(list(array), # MappedArray .tolist() does not output states.
                         ['Stowed']*10+['Armed/Cmd Dn']*10+['Deployed/Cmd Up']*10+['Stowed']*10)

        handle_array = np.ma.concatenate([np.ma.arange(0, 2, 0.1),
                                          np.ma.array([13]*10),
                                          np.ma.arange(0.99, 0, -0.1)])
        spd_sel = SpeedbrakeSelected()
        array = spd_sel.derive_from_handle(handle_array, deployed=5)
        self.assertEqual(list(array), # MappedArray .tolist() does not output states.
                         ['Stowed']*20+['Deployed/Cmd Up']*10+['Stowed']*10)
    
    def test_derive_from_handle_mask_below_armed(self):
        array = np.ma.arange(-5, 15)
        result = SpeedbrakeSelected.derive_from_handle(array, deployed=10,
                                                       armed=0,
                                                       mask_below_armed=True)
        self.assertEqual(list(result),
                         [np.ma.masked] * 5 + ['Stowed'] + 9 * ['Armed/Cmd Dn'] +
                         5 * ['Deployed/Cmd Up'])

    def test_bd100_speedbrake(self):
        handle = load(os.path.join(
            test_data_path, 'SpeedbrakeSelected_SpeedbrakeHandle_2.nod'))
        spoiler_gnd_armed = load(os.path.join(
            test_data_path, 'SpeedbrakeSelected_SpoilerGroundArmed_2.nod'))
        array = SpeedbrakeSelected.bd100_speedbrake(handle.array,
                                                    spoiler_gnd_armed.array)
        self.assertTrue(all(x == 'Armed/Cmd Dn' for x in
                            array[spoiler_gnd_armed.array == 'Armed']))
        self.assertEqual(np.ma.concatenate([np.ma.arange(8802, 8824),
                                            np.ma.arange(11463, 11523),
                                            np.ma.arange(11545, 11575),
                                            np.ma.arange(11840, 12013)]).tolist(),
                         np.ma.where(array == 'Deployed/Cmd Up')[0].tolist())

    def test_b737_speedbrake(self):
        self.maxDiff = None
        spd_sel = SpeedbrakeSelected()
        spdbrk = P(array=np.ma.array([0]*10 + [1.3]*20 + [0.2]*10))
        handle = P(array=np.ma.arange(40))
        # Follow the spdbrk only
        res = spd_sel.b737_speedbrake(spdbrk, None)
        self.assertEqual(list(res),
                        ['Stowed']*10 + ['Deployed/Cmd Up']*20 + ['Stowed']*10)
        # Follow the handle only
        res = spd_sel.b737_speedbrake(None, handle)
        self.assertEqual(list(res),
                        ['Stowed']*3 + ['Armed/Cmd Dn']*32 + ['Deployed/Cmd Up']*5)
        # Follow the combination
        res = spd_sel.b737_speedbrake(spdbrk, handle)
        self.assertEqual(list(res),
                        ['Stowed']*3 + ['Armed/Cmd Dn']*7 + ['Deployed/Cmd Up']*20 + ['Armed/Cmd Dn']*5 + ['Deployed/Cmd Up']*5)

    def test_b787_speedbrake(self):
        handle = load(os.path.join(
            test_data_path, 'SpeedBrakeSelected_SpeedbrakeHandle.nod'))

        result = SpeedbrakeSelected.b787_speedbrake(handle)
        self.assertEqual(len(np.ma.where(result == 0)[0]), 9445)
        self.assertEqual(np.ma.where(result == 1)[0].tolist(),
                         [8189, 8190, 8451, 8524, 8525] + range(9098, 9223))
        self.assertEqual(np.ma.where(result == 2)[0].tolist(),
                         range(8191, 8329) + range(8452, 8524) + range(9223, 9262))


class TestStableApproach(unittest.TestCase):
    def test_can_operate(self):
        opts = StableApproach.get_operational_combinations()
        combinations = [
            # all
            ('Approach And Landing', 'Gear Down', 'Flap', 'Track Deviation From Runway', 'Airspeed Relative For 3 Sec', 'Vertical Speed', 'ILS Glideslope', 'ILS Localizer', 'Eng (*) N1 Min For 5 Sec', 'Altitude AAL', 'Vapp'),
            # exc. Vapp
            ('Approach And Landing', 'Gear Down', 'Flap', 'Track Deviation From Runway', 'Airspeed Relative For 3 Sec', 'Vertical Speed', 'ILS Glideslope', 'ILS Localizer', 'Eng (*) N1 Min For 5 Sec', 'Altitude AAL'),
            # exc. Airspeed Relative
            ('Approach And Landing', 'Gear Down', 'Flap', 'Track Deviation From Runway', 'Vertical Speed', 'ILS Glideslope', 'ILS Localizer', 'Eng (*) N1 Min For 5 Sec', 'Altitude AAL', 'Vapp'),
            # exc. Vapp and Airspeed Relative
            ('Approach And Landing', 'Gear Down', 'Flap', 'Track Deviation From Runway', 'Vertical Speed', 'ILS Glideslope', 'ILS Localizer', 'Eng (*) N1 Min For 5 Sec', 'Altitude AAL'),
            # exc. ILS Glideslope and Vapp
            ('Approach And Landing', 'Gear Down', 'Flap', 'Track Deviation From Runway', 'Airspeed Relative For 3 Sec', 'Vertical Speed', 'ILS Localizer', 'Eng (*) N1 Min For 5 Sec', 'Altitude AAL'),
            # exc. ILS Glideslope and ILS Localizer and Vapp
            ('Approach And Landing', 'Gear Down', 'Flap', 'Track Deviation From Runway', 'Airspeed Relative For 3 Sec', 'Vertical Speed', 'Eng (*) N1 Min For 5 Sec', 'Altitude AAL'),
            # using EPR and exc. Airspeed Relative
            ('Approach And Landing', 'Gear Down', 'Flap', 'Track Deviation From Runway', 'Vertical Speed', 'ILS Glideslope', 'ILS Localizer', 'Eng (*) EPR Min For 5 Sec', 'Altitude AAL', 'Vapp'),
            # including Family attribute
            ('Approach And Landing', 'Gear Down', 'Flap', 'Track Deviation From Runway', 'Vertical Speed', 'ILS Glideslope', 'ILS Localizer', 'Eng (*) EPR Min For 5 Sec', 'Altitude AAL', 'Vapp', 'Family'),
        ]
        for combo in combinations:
            self.assertIn(combo, opts)

    def test_stable_approach(self):
        stable = StableApproach()

        # Arrays will be 20 seconds long, index 4, 13,14,15 are stable
        #0. first and last values are not in approach slice
        apps = S()
        apps.create_section(slice(1,20))
        #1. gear up for index 0-2
        g = [ 0,  0,  0,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1]
        gear = M(array=np.ma.array(g), values_mapping={1:'Down'})
        #2. landing flap invalid index 0, 5
        f = [ 5, 15, 15, 15, 15,  0, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15]
        flap = P(array=np.ma.array(f))
        #3. Heading stays within limits except for index 11-12, although we weren't on the heading sample 15 (masked out)
        h = [20, 20,  2,  3,  4,  8,  0,  0,  0,  0,  2, 20, 20,  8,  2,  0,  1,  1,  1,  1,  1]
        hm= [ 1,  1,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0]
        head = P(array=np.ma.array(h, mask=hm))
        #4. airspeed relative within limits for periods except 0-3
        a = [50, 50, 50, 45,  9,  8,  3, 7,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0]
        aspd = P(array=np.ma.array(a))
        #5. glideslope deviation is out for index 8, index 10-11 ignored as under 200ft, last 4 values ignored due to alt cutoff
        g = [ 6,  6,  6,  6,  0, .5, .5,-.5,1.2,0.9,1.4,1.3,  0,  0,  0,  0,  0, -2, -2, -2, -2]
        gm= [ 1,  1,  1,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0]
        glide = P(array=np.ma.array(g, mask=gm))
        #6. localizer deviation is out for index 7, index 10 ignored as just under 200ft, last 4 values ignored due to alt cutoff
        l = [ 0,  0,  0,  0,  0,  0,  0,  2,  0.8, 0.1, -3,  0,  0,  0,  0,  0,  0, -2, -2, -2, -2]
        loc = P(array=np.ma.array(l))
        #7. Vertical Speed too great at index 8, but change is smoothed out and at 17 (59ft)
        v = [-500] * 20
        v[6] = -2000
        v[18:19] = [-2000]*1
        vert_spd = P(array=np.ma.array(v))

        #TODO: engine cycling at index 12?

        #8. Engine power too low at index 5-12
        e = [80, 80, 80, 80, 80, 30, 20, 30, 20, 30, 20, 30, 44, 40, 80, 80, 80, 50, 50, 50, 50]
        eng = P(array=np.ma.array(e))

        # Altitude for cutoff heights, 9th element is 200 below, last 4 values are below 100ft last 2 below 50ft
        al = range(2000,219,-200) + range(219,18, -20) + [0]
        # == [2000, 1800, 1600, 1400, 1200, 1000, 800, 600, 400, 219, 199, 179, 159, 139, 119, 99, 79, 59, 39, 19]
        alt = P(array=np.ma.array(al))
        # DERIVE without using Vapp (using Vref limits)
        stable.derive(apps, gear, flap, head, aspd, vert_spd, glide, loc, eng, None, alt, None)
        self.assertEqual(len(stable.array), len(alt.array))
        self.assertEqual(len(stable.array), len(head.array))

        self.assertEqual(list(stable.array.data),
        #index: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,16,17,18,19,20
               [0, 1, 1, 4, 9, 2, 8, 6, 5, 8, 8, 3, 3, 8, 9, 9, 9, 9, 9, 9, 0])
        self.assertEqual(list(stable.array.mask),
               [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])

        #========== NO GLIDESLOPE ==========
        # Test without the use of Glideslope (not on it at 1000ft) therefore
        # instability for index 7-10 is now due to low Engine Power
        glide2 = P(array=np.ma.array([3.5]*20))
        stable.derive(apps, gear, flap, head, aspd, vert_spd, glide2, loc, eng, None, alt, None)
        self.assertEqual(list(stable.array.data),
        #index: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,16,17,18,19,20
               [0, 1, 1, 4, 9, 2, 8, 8, 8, 8, 8, 3, 3, 8, 9, 9, 9, 9, 9, 9, 0])

        #========== VERTICAL SPEED ==========
        # Test with a lot of vertical speed (rather than just gusts above)
        v2 = [-1800] * 20
        vert_spd2 = P(array=np.ma.array(v2))
        stable.derive(apps, gear, flap, head, aspd, vert_spd2, glide2, loc, eng, None, alt, None)
        self.assertEqual(list(stable.array.data),
        #index: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,16,17,18,19,20
               [0, 1, 1, 4, 9, 2, 7, 7, 7, 7, 7, 3, 3, 7, 7, 7, 9, 9, 9, 9, 0])

        #========== UNSTABLE GLIDESLOPE JUST ABOVE 200ft ==========
        # Test that with unstable glideslope just before 200ft, this stability
        # reason is continued to touchdown. Higher level checks (Heading at 3)
        # still take priority at indexes 11-12
        #                                        219ft == 1.5 dots
        g3 = [ 6,  6,  6,  6,  0, .5, .5,-.5,1.2,1.5,1.4,1.3,  0,  0,  0,  0,  0, -2, -2, -2, -2]
        gm = [ 1,  1,  1,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0]
        glide3 = P(array=np.ma.array(g3, mask=gm))
        stable.derive(apps, gear, flap, head, aspd, vert_spd, glide3, loc, eng, None, alt, None)
        self.assertEqual(list(stable.array.data),
        #index: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,16,17,18,19,20
               [0, 1, 1, 4, 9, 2, 8, 6, 5, 5, 5, 3, 3, 5, 5, 5, 5, 5, 5, 5, 0])


    def test_with_real_data(self):

        apps = S(items=[Section(name='Approach And Landing',
                                slice=slice(2702, 2993, None),
                                start_edge=2702.0, stop_edge=2993.0)])

        #def save_test_node(param):
            #param.save('../tests/test_data/Stable Approach - '+param.name+'.nod')

        def test_node(name):
            return load(os.path.join(test_data_path, 'Stable Approach - '+name+'.nod'))
        stable = StableApproach()

        gear = test_node('Gear Down')
        flap = test_node('Flap')
        tdev = test_node('Track Deviation From Runway')
        vspd = test_node('Vertical Speed')
        gdev = test_node('ILS Glideslope')
        ldev = test_node('ILS Localizer')
        eng = test_node('Eng (star) N1 Min For 5 Sec')
        alt = test_node('Altitude AAL')

        stable.derive(
            apps=apps,
            gear=gear,
            flap=flap,
            tdev=tdev,
            aspd=None,
            vspd=vspd,
            gdev=gdev,
            ldev=ldev,
            eng_n1=eng,
            eng_epr=None,
            alt=alt,
            vapp=None)

        self.assertEqual(len(stable.array), len(alt.array))
        analysed = np.ma.clump_unmasked(stable.array)
        self.assertEqual(len(analysed), 1)
        # valid for the approach slice
        self.assertEqual(analysed[0].start, apps[0].slice.start)
        # stop is 10 secs after touchdown
        self.assertEqual(analysed[0].stop, 2946)

        sect = stable.array[analysed[0]]
        # assert that last few values are correct (masked in gear and flap params should not influence end result)
        self.assertEqual(list(sect[-4:]), ['Stable']*4)
        self.assertEqual(list(sect[0:73]), ['Gear Not Down']*73)
        self.assertEqual(list(sect[74:117]), ['Not Landing Flap']*43)
        self.assertTrue(np.all(sect[117:] == ['Stable']))


class TestStickShaker(unittest.TestCase):

    def test_can_operate(self):
        opts = StickShaker.get_operational_combinations()
        self.assertEqual(len(opts), 126)

    def test_derive(self):
        left = M('Stick Shaker (L)', np.ma.array([0, 1, 0, 0, 0, 0]),
                 offset=0.7, frequency=2.0,
                 values_mapping={0: '-', 1: 'Shake'})
        right = M('Stick Shaker (R)', np.ma.array([0, 0, 0, 0, 1, 0]),
                  offset=0.2, frequency=2.0,
                  values_mapping={0: '-', 1: 'Shake'})
        ss = StickShaker()
        ss.derive(left, right, None, None, None, None)
        expected = np.ma.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0])
        np.testing.assert_equal(ss.array.raw, expected)

    def test_single_source(self):
        left=M('Stick Shaker (L)',np.ma.array([0,1,0,0,0,0]),
               offset=0.7, frequency=2.0,
               values_mapping = {0: '-',1: 'Shake',})
        ss=StickShaker()
        ss.derive(left, None, None, None, None, None)
        expected = np.ma.array([0,1,0,0,0,0])
        np.testing.assert_equal(ss.array, expected)

    def test_not_777(self):
        left=M('Stick Shaker (L)',np.ma.array([0,1,0,0,0,0]),
                       offset=0.7, frequency=2.0,
                       values_mapping = {0: '-',1: 'Shake',})
        ss=StickShaker()
        self.assertRaises(ValueError, ss.derive,
                          left, None, None, None, None, None,
                          A('Frame', 'B777'))

class TestStickPusher(unittest.TestCase):

    def test_can_operate(self):
        opts = StickPusher.get_operational_combinations()
        self.assertEqual(len(opts), 3)

    def test_derive(self):
        left = M('Stick Pusher (L)', np.ma.array([0, 1, 0, 0, 0, 0]),
                 offset=0.7, frequency=2.0,
                 values_mapping={0: '-', 1: 'Shake'})
        right = M('Stick Pusher (R)', np.ma.array([0, 0, 0, 0, 1, 0]),
                  offset=0.2, frequency=2.0,
                  values_mapping={0: '-', 1: 'Shake'})
        sp = StickPusher()
        sp.derive(left, right)
        expected = np.ma.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0])
        np.testing.assert_equal(sp.array, expected)

    def test_single_source(self):
        left = M('Stick Pusher (L)',np.ma.array([0,1,0,0,0,0]),
                 offset=0.7, frequency=2.0,
                 values_mapping = {0: '-',1: 'Push',})
        sp = StickPusher()
        sp.derive(None, left) # Just for variety
        expected = np.ma.array([0,1,0,0,0,0])
        np.testing.assert_equal(sp.array, expected)


class TestThrustReversers(unittest.TestCase):

    @unittest.skip('Test Not Implemented')
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')

    def setUp(self):
        eng_1_unlocked_array = [ 1,  1,  1,  1,  0,  0,  0,  0,  0,  0]
        eng_1_deployed_array = [ 1,  1,  1,  1,  0,  0,  0,  0,  0,  0]
        eng_2_unlocked_array = [ 1,  1,  1,  1,  0,  1,  0,  0,  0,  0]
        eng_2_deployed_array = [ 1,  1,  1,  1,  1,  0,  0,  0,  0,  0]

        self.eng_1_unlocked = M(name='Eng (1) Thrust Reverser Unlocked', array=np.ma.array(eng_1_unlocked_array), values_mapping={1:'Unlocked'})
        self.eng_1_deployed = M(name='Eng (1) Thrust Reverser Deployed', array=np.ma.array(eng_1_deployed_array), values_mapping={1:'Deployed'})
        self.eng_2_unlocked = M(name='Eng (2) Thrust Reverser Unlocked', array=np.ma.array(eng_2_unlocked_array), values_mapping={1:'Unlocked'})
        self.eng_2_deployed = M(name='Eng (2) Thrust Reverser Deployed', array=np.ma.array(eng_2_deployed_array), values_mapping={1:'Deployed'})
        self.thrust_reversers = ThrustReversers()

    def test_derive(self):
        result = [ 2,  2,  2,  2,  1,  1,  0,  0,  0,  0]
        self.thrust_reversers.get_derived([self.eng_1_deployed,
                                None,
                                None,
                                self.eng_1_unlocked,
                                None,
                                None,
                                None,
                                self.eng_2_deployed,
                                None,
                                None,
                                self.eng_2_unlocked] + [None] * 17)
        np.testing.assert_equal(self.thrust_reversers.array.data, result)

    def test_derive_masked_value(self):
        self.eng_1_unlocked.array.mask = [ 0,  0,  0,  0,  0,  1,  0,  0,  1,  0]
        self.eng_1_deployed.array.mask = [ 0,  0,  0,  1,  0,  1,  0,  0,  1,  0]
        self.eng_2_unlocked.array.mask = [ 0,  0,  0,  1,  0,  0,  0,  1,  1,  0]
        self.eng_2_deployed.array.mask = [ 0,  0,  0,  0,  0,  1,  0,  0,  1,  0]

        result_array = [ 2,  2,  2,  2,  1,  1,  0,  0,  0,  0]
        result_mask =  [ 0,  0,  0,  0,  0,  1,  0,  0,  1,  0]

        self.thrust_reversers.get_derived([self.eng_1_deployed,
                                None,
                                None,
                                self.eng_1_unlocked,
                                None,
                                None,
                                None,
                                self.eng_2_deployed,
                                None,
                                None,
                                self.eng_2_unlocked] + [None] * 17)
        np.testing.assert_equal(self.thrust_reversers.array.data, result_array)
        np.testing.assert_equal(self.thrust_reversers.array.mask, result_mask)

    def test_derive_in_transit_avaliable(self):
        result = [ 2,  2,  1,  1,  1,  1,  0,  0,  0,  0]
        transit_array = [ 0,  0,  1,  1,  1,  1,  0,  0,  0,  0]
        eng_1_in_transit = M(name='Eng (1) Thrust Reverser In Transit', array=np.ma.array(transit_array), values_mapping={1:'In Transit'})
        self.thrust_reversers.get_derived([self.eng_1_deployed,
                                None,
                                None,
                                self.eng_1_unlocked,
                                None,
                                None,
                                eng_1_in_transit,
                                self.eng_2_deployed,
                                None,
                                None,
                                self.eng_2_unlocked] + [None] * 17)
        np.testing.assert_equal(self.thrust_reversers.array.data, result)

    def test_derive_unlock_at_edges(self):
        '''
        test for aircraft which only record Thrust Reverser Unlocked during
        transition, not whilst deployed
        '''
        result =               [ 0, 1, 1, 1, 2, 2, 1, 1, 0, 0]

        eng_1_unlocked_array = [ 0, 1, 1, 0, 0, 0, 1, 1, 0, 0]
        eng_1_deployed_array = [ 0, 0, 0, 1, 1, 1, 0, 0, 0, 0]
        eng_2_unlocked_array = [ 0, 0, 1, 1, 0, 0, 1, 1, 0, 0]
        eng_2_deployed_array = [ 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]

        eng_1_unlocked = M(name='Eng (1) Thrust Reverser Unlocked', array=np.ma.array(eng_1_unlocked_array), values_mapping={1:'Unlocked'})
        eng_1_deployed = M(name='Eng (1) Thrust Reverser Deployed', array=np.ma.array(eng_1_deployed_array), values_mapping={1:'Deployed'})
        eng_2_unlocked = M(name='Eng (2) Thrust Reverser Unlocked', array=np.ma.array(eng_2_unlocked_array), values_mapping={1:'Unlocked'})
        eng_2_deployed = M(name='Eng (2) Thrust Reverser Deployed', array=np.ma.array(eng_2_deployed_array), values_mapping={1:'Deployed'})

        self.thrust_reversers.get_derived([eng_1_deployed,
                                None,
                                None,
                                eng_1_unlocked,
                                None,
                                None,
                                None,
                                eng_2_deployed,
                                None,
                                None,
                                eng_2_unlocked] + [None] * 17)
        np.testing.assert_equal(self.thrust_reversers.array.data, result)


class TestTakeoffConfigurationWarning(unittest.TestCase):

    def test_can_operate(self):
        self.assertTrue(TakeoffConfigurationWarning.can_operate(
            ('Takeoff Configuration Stabilizer Warning',)))
        self.assertTrue(TakeoffConfigurationWarning.can_operate(
            ('Takeoff Configuration Parking Brake Warning',)))
        self.assertTrue(TakeoffConfigurationWarning.can_operate(
            ('Takeoff Configuration Flap Warning',)))
        self.assertTrue(TakeoffConfigurationWarning.can_operate(
            ('Takeoff Configuration Gear Warning',)))
        self.assertTrue(TakeoffConfigurationWarning.can_operate(
            ('Takeoff Configuration AP Warning',)))
        self.assertTrue(TakeoffConfigurationWarning.can_operate(
            ('Takeoff Configuration Aileron Warning',)))
        self.assertTrue(TakeoffConfigurationWarning.can_operate(
            ('Takeoff Configuration Rudder Warning',)))
        self.assertTrue(TakeoffConfigurationWarning.can_operate(
            ('Takeoff Configuration Spoiler Warning',)))
        self.assertTrue(TakeoffConfigurationWarning.can_operate(
            ('Takeoff Configuration Stabilizer Warning',
             'Takeoff Configuration Parking Brake Warning',
             'Takeoff Configuration Flap Warning',
             'Takeoff Configuration Gear Warning',
             'Takeoff Configuration Rudder Warning',
             'Takeoff Configuration Spoiler Warning',)))

    @unittest.skip('Test Not Implemented')
    def test_derive_basic(self):
        pass


class TestTAWSAlert(unittest.TestCase):
    def test_can_operate(self):
        parameters = ['TAWS Caution Terrain',
                       'TAWS Caution',
                       'TAWS Dont Sink',
                       'TAWS Glideslope'
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
                       ]
        for p in parameters:
            self.assertTrue(TAWSAlert.can_operate(p))

    def setUp(self):
        terrain_array = [1,1,0,1,1,0,0,0,1,0,1,0,0,0,0,0,1,0,0,0]
        pull_up_array = [0,1,1,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,1,0]

        self.airs = S(name='Airborne')
        self.airs.create_section(slice(5,15))
        self.terrain = M(name='TAWS Terrain', array=np.ma.array(terrain_array), values_mapping={1:'Warning'})
        self.pull_up = M(name='TAWS Pull Up', array=np.ma.array(pull_up_array), values_mapping={1:'Warning'})
        self.taws_alert = TAWSAlert()

    def test_derive(self):
        result = [0,0,0,0,0,0,0,0,1,0,1,1,1,0,0,0,0,0,0,0]

        self.taws_alert.get_derived((self.airs,
                                None,
                                None,
                                None,
                                None,
                                None,
                                self.pull_up,
                                None,
                                None,
                                None,
                                None,
                                self.terrain,
                                None,
                                None,
                                None,
                                None,))
        np.testing.assert_equal(self.taws_alert.array.data, result)

    def test_derive_masked_values(self):
        result = [0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0]
        self.terrain.array[8] = np.ma.masked
        self.terrain.array[10] = np.ma.masked

        self.taws_alert.get_derived((self.airs,
                                None,
                                None,
                                None,
                                None,
                                None,
                                self.pull_up,
                                None,
                                None,
                                None,
                                None,
                                self.terrain,
                                None,
                                None,
                                None,
                                None,))
        np.testing.assert_equal(self.taws_alert.array.data, result)

    def test_derive_zeros(self):
        result = [0,0,0,0,0,0,0,0,1,0,1,1,1,0,0,0,0,0,0,0]

        terrain_array = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

        caution = M(name='TAWS Caution Terrain', array=np.ma.array(terrain_array), values_mapping={1:'Warning'})
        caution.array.mask = True

        self.taws_alert.get_derived((self.airs,
                                     caution,
                                     None,
                                     None,
                                     None,
                                     None,
                                     self.pull_up,
                                     None,
                                     None,
                                     None,
                                     None,
                                     self.terrain,
                                     None,
                                     None,
                                     None,
                                     None,))
        np.testing.assert_equal(self.taws_alert.array.data, result)


class TestTAWSDontSink(unittest.TestCase):

    def test_can_operate(self):
        self.assertEqual(TAWSDontSink.get_operational_combinations(),
                         [('TAWS (L) Dont Sink',),
                          ('TAWS (R) Dont Sink',),
                          ('TAWS (L) Dont Sink', 'TAWS (R) Dont Sink')])

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        pass


class TestTAWSGlideslopeCancel(unittest.TestCase):

    def test_can_operate(self):
        self.assertEqual(TAWSGlideslopeCancel.get_operational_combinations(),
                         [('TAWS (L) Glideslope Cancel',),
                          ('TAWS (R) Glideslope Cancel',),
                          ('TAWS (L) Glideslope Cancel', 'TAWS (R) Glideslope Cancel')])

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        pass


class TestTAWSTooLowGear(unittest.TestCase):

    def test_can_operate(self):
        self.assertEqual(TAWSTooLowGear.get_operational_combinations(),
                         [('TAWS (L) Too Low Gear',),
                          ('TAWS (R) Too Low Gear',),
                          ('TAWS (L) Too Low Gear', 'TAWS (R) Too Low Gear')])

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        pass


class TestTCASFailure(unittest.TestCase):

    def test_can_operate(self):
        self.assertEqual(TCASFailure.get_operational_combinations(),
                         [('TCAS (L) Failure',),
                          ('TCAS (R) Failure',),
                          ('TCAS (L) Failure', 'TCAS (R) Failure')])

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        pass


class TestSpeedControl(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = SpeedControl
        self.operational_combinations = [
            ('Speed Control Auto',),
            ('Speed Control Manual',),
            ('Speed Control Auto', 'Speed Control Manual'),
            ('Speed Control (1) Auto',),
            ('Speed Control (1) Manual',),
            ('Speed Control (1) Auto', 'Speed Control (1) Manual'),
            ('Speed Control (2) Auto',),
            ('Speed Control (2) Manual',),
            ('Speed Control (2) Auto', 'Speed Control (2) Manual'),
        ]

    def test_derive(self):
        sc0a = M(
            name='Speed Control Auto',
            array=np.ma.array((0, 0, 1, 1, 0, 0)),
            values_mapping={0: 'Manual', 1: 'Auto'},
        )
        sc0m = M(
            name='Speed Control Manual',
            array=np.ma.array((0, 1, 1, 1, 1, 0)),
            values_mapping={0: 'Auto', 1: 'Manual'},
        )
        node = self.node_class()
        node.derive(sc0a, sc0m, None, None, None, None)
        expected = M(
            name='Speed Control',
            array=np.ma.array((1, 0, 1, 1, 0, 1)),
            values_mapping={0: 'Manual', 1: 'Auto'},
        )
        ma_test.assert_array_equal(node.array, expected.array)
