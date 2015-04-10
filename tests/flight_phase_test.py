import numpy as np
import os
import unittest

from flightdatautilities.array_operations import load_compressed
from flightdatautilities.filesystem_tools import copy_file

from analysis_engine.flight_phase import (
    Airborne,
    Approach,
    ApproachAndLanding,
    BouncedLanding,
    ClimbCruiseDescent,
    Climbing,
    Cruise,
    Descending,
    Descent,
    DescentLowClimb,
    DescentToFlare,
    EngHotelMode,
    Fast,
    FinalApproach,
    GearExtended,
    GearExtending,
    GearRetracted,
    GearRetracting,
    GoAroundAndClimbout,
    GoAround5MinRating,
    Grounded,
    Holding,
    ILSGlideslopeEstablished,
    ILSLocalizerEstablished,
    InitialApproach,
    InitialClimb,
    InitialCruise,
    Landing,
    LandingRoll,
    LevelFlight,
    Mobile,
    RejectedTakeoff,
    StraightAndLevel,
    Takeoff,
    Takeoff5MinRating,
    TakeoffRoll,
    TakeoffRollOrRejectedTakeoff,
    TakeoffRotation,
    Taxiing,
    TaxiIn,
    TaxiOut,
    TurningInAir,
    TurningOnGround,
    TwoDegPitchTo35Ft,
)
from analysis_engine.key_time_instances import BottomOfDescent, TopOfClimb, TopOfDescent
from analysis_engine.library import integrate
from analysis_engine.node import (Attribute, KTI, KeyTimeInstance, M, Parameter,
                                  P, S, Section, SectionNode, load)
from analysis_engine.process_flight import process_flight

from analysis_engine.settings import AIRSPEED_THRESHOLD


test_data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              'test_data')

'''
Three little routines to make building Sections for testing easier.
'''
def builditem(name, begin, end):
    '''
    This code more accurately represents the aligned section values, but is
    not suitable for test cases where the data does not get aligned.

    if begin is None:
        ib = None
    else:
        ib = int(begin)
        if ib < begin:
            ib += 1
    if end is None:
        ie = None
    else:
        ie = int(end)
        if ie < end:
            ie += 1
            '''
    slice_end = end if end is None else end + 1
    return Section(name, slice(begin, slice_end, None), begin, end)


def buildsection(name, begin, end):
    '''
    A little routine to make building Sections for testing easier.

    :param name: name for a test Section
    :param begin: index at start of section
    :param end: index at end of section

    :returns: a SectionNode populated correctly.

    Example: land = buildsection('Landing', 100, 120)
    '''
    result = builditem(name, begin, end)
    return SectionNode(name, items=[result])


def buildsections(*args):
    '''
    Like buildsection, this is used to build SectionNodes for test purposes.

    lands = buildsections('name',[from1,to1],[from2,to2])

    Example of use:
    approach = buildsections('Approach', [80,90], [100,110])
    '''
    built_list=[]
    name = args[0]
    for a in args[1:]:
        new_section = builditem(name, a[0], a[1])
        built_list.append(new_section)
    return SectionNode(name, items=built_list)

def build_kti(*args):
    built_list=[]
    name = args[0]
    for a in args[1:]:
        if a:
            new_kti = KeyTimeInstance(a, name)
            built_list.append(new_kti)
    return KTI(items=built_list)


##############################################################################
# Superclasses


class NodeTest(object):

    def test_can_operate(self):
        self.assertEqual(
            self.node_class.get_operational_combinations(),
            self.operational_combinations,
        )


##############################################################################


class TestAirborne(unittest.TestCase):
    # Based closely on the level flight condition, but taking only the
    # outside edges of the envelope.
    def test_can_operate(self):
        node = Airborne
        expected = ('Altitude AAL For Flight Phases', 'Fast')
        self.assertTrue(node.can_operate(expected, seg_type=Attribute('Segment Type', 'START_AND_STOP')))
        self.assertFalse(node.can_operate(expected, seg_type=Attribute('Segment Type', 'GROUND_ONLY')))

    def test_airborne_phase_basic(self):
        # First sample with altitude more than zero is 6, last with high speed is 80.
        vert_spd_data = np.ma.array([0] * 5 + range(0,400,20)+
                                    range(400,-400,-20)+
                                    range(-400,50,20))
        altitude = Parameter('Altitude AAL For Flight Phases', integrate(vert_spd_data, 1, 0, 1.0/60.0))
        fast = SectionNode('Fast', items=[Section(name='Airborne', slice=slice(3, 80, None), start_edge=3, stop_edge=80)])
        air = Airborne()
        air.derive(altitude, fast)
        expected = [Section(name='Airborne', slice=slice(6, 80, None), start_edge=6, stop_edge=80)]
        self.assertEqual(list(air), expected)

    def test_airborne_phase_not_fast(self):
        altitude_data = np.ma.array(range(0,10))
        alt_aal = Parameter('Altitude AAL For Flight Phases', altitude_data)
        fast = SectionNode('Fast')
        air = Airborne()
        air.derive(alt_aal, fast)
        self.assertEqual(air, [])

    @unittest.skip('TODO: Test to be amended')
    def test_airborne_phase_started_midflight(self):
        altitude_data = np.ma.array([100]*20+[60,30,10]+[0]*4)
        alt_aal = Parameter('Altitude AAL For Flight Phases', altitude_data)
        fast = buildsection('Fast', None, 25)
        air = Airborne()
        air.derive(alt_aal, fast)
        # The problem here is that buildsection now returns a slice to 24 and
        # a stop_edge to 23. Probably not worth fixing this test if we are
        # going to turn over to segments.
        expected = buildsection('Airborne', None, 23)
        self.assertEqual(air, expected)

    def test_airborne_phase_ends_in_midflight(self):
        altitude_data = np.ma.array([0]*5+[30,80]+[100]*20)
        alt_aal = Parameter('Altitude AAL For Flight Phases', altitude_data)
        fast = buildsection('Fast', 2, None)
        air = Airborne()
        air.derive(alt_aal, fast)
        expected = buildsection('Airborne', 5, None)
        self.assertEqual(list(air), list(expected))
        
    def test_airborne_fast_with_gaps(self):
        alt_aal = P('Altitude AAL For Flight Phases',
                    np.ma.array([10000]*60),frequency=0.1)
        fast = buildsections('Fast', [1,10],[15,24],[30,36],[40,50],[55,59])
        fast.frequency = 0.1
        air = Airborne()
        air.derive(alt_aal, fast)
        self.assertEqual(len(air), 2)
        self.assertEqual(air[0].slice.start, 1)
        self.assertEqual(air[0].slice.stop, 24)
        self.assertEqual(air[1].slice.start, 30)
        self.assertEqual(air[1].slice.stop, 59)


class TestApproachAndLanding(unittest.TestCase):
    def test_can_operate(self):
        node = ApproachAndLanding
        start_stop = Attribute('Segment Type', 'START_AND_STOP')
        ground_only = Attribute('Segment Type', 'GROUND_ONLY')
        self.assertTrue(node.can_operate(('Altitude AAL For Flight Phases', 'Level Flight', 'Landing',),
                                             seg_type=start_stop))
        self.assertTrue(node.can_operate(('Altitude AAL For Flight Phases',),
                                             seg_type=start_stop))
        self.assertTrue(node.can_operate(('Altitude AAL For Flight Phases', 'Landing'),
                                             seg_type=start_stop))
        self.assertTrue(node.can_operate(('Altitude AAL For Flight Phases', 'Level Flight'),
                                             seg_type=start_stop))
        
        self.assertFalse(node.can_operate(('Altitude AAL For Flight Phases', 'Level Flight'),
                                              seg_type=ground_only))

    def test_approach_and_landing_basic(self):
        alt = np.ma.array(range(5000, 500, -500) + [0] * 10)
        # No Go-arounds detected
        gas = KTI(items=[])
        app = ApproachAndLanding()
        app.derive(Parameter('Altitude AAL For Flight Phases', alt), None, None)
        self.assertEqual(app.get_slices(), [slice(4.0, 9)])

    def test_go_around_below_1500ft(self):
        alt_aal = load(os.path.join(test_data_path,
                                    'GoAroundAndClimbout_alt_aal.nod'))
        app_ldg = ApproachAndLanding()
        app_ldg.derive(alt_aal, None, None)
        self.assertEqual(len(app_ldg), 6)
        self.assertAlmostEqual(app_ldg[0].slice.start, 1005, places=0)
        self.assertAlmostEqual(app_ldg[0].slice.stop, 1111, places=0)
        self.assertAlmostEqual(app_ldg[1].slice.start, 1378, places=0)
        self.assertAlmostEqual(app_ldg[1].slice.stop, 1458, places=0)
        self.assertAlmostEqual(app_ldg[2].slice.start, 1676, places=0)
        self.assertAlmostEqual(app_ldg[2].slice.stop, 1783, places=0)
        self.assertAlmostEqual(app_ldg[3].slice.start, 2021, places=0)
        self.assertAlmostEqual(app_ldg[3].slice.stop, 2116, places=0)
        self.assertAlmostEqual(app_ldg[4].slice.start, 2208, places=0)
        self.assertAlmostEqual(app_ldg[4].slice.stop, 2468, places=0)
        self.assertAlmostEqual(app_ldg[5].slice.start, 2680, places=0)
        self.assertAlmostEqual(app_ldg[5].slice.stop, 2806, places=0)
        
    def test_go_around_2(self):
        alt_aal = load(os.path.join(test_data_path, 'alt_aal_goaround.nod'))
        level_flights = SectionNode('Level Flight')
        level_flights.create_sections([
            slice(1629.0, 2299.0, None),
            slice(3722.0, 4708.0, None),
            slice(4726.0, 4805.0, None),
            slice(5009.0, 5071.0, None),
            slice(5168.0, 6883.0, None),
            slice(8433.0, 9058.0, None)])
        landings = buildsection('Landing', 10500, 10749)
        app_ldg = ApproachAndLanding()
        app_ldg.derive(alt_aal, level_flights, landings)
        self.assertEqual(len(app_ldg), 4)
        self.assertAlmostEqual(app_ldg[0].slice.start, 3425, places=0)
        self.assertAlmostEqual(app_ldg[0].slice.stop, 3632, places=0)
        self.assertAlmostEqual(app_ldg[1].slice.start, 4805, places=0)
        self.assertAlmostEqual(app_ldg[1].slice.stop, 4941, places=0)
        self.assertAlmostEqual(app_ldg[2].slice.start, 6883, places=0)
        self.assertAlmostEqual(app_ldg[2].slice.stop, 7171, places=0)
        self.assertAlmostEqual(app_ldg[3].slice.start, 10362, places=0)
        self.assertAlmostEqual(app_ldg[3].slice.stop, 10750, places=0)

        
    def test_with_go_around_and_climbout_atr42_data(self):
        alt_aal = load(os.path.join(test_data_path,
                                    'AltitudeAAL_ATR42_two_goarounds.nod'))
        app_ldg = ApproachAndLanding()
        app_ldg.derive(alt_aal, None, None)
        self.assertEqual(len(app_ldg), 3)
        self.assertAlmostEqual(app_ldg[0].slice.start, 9771, places=0)
        self.assertAlmostEqual(app_ldg[0].slice.stop, 10810, places=0)
        self.assertAlmostEqual(app_ldg[1].slice.start, 12050, places=0)
        self.assertAlmostEqual(app_ldg[1].slice.stop, 12631, places=0)
        self.assertAlmostEqual(app_ldg[2].slice.start, 26926, places=0)
        self.assertAlmostEqual(app_ldg[2].slice.stop, 27359, places=0)
    
    @unittest.skip('Algorithm does not successfully this noisy signal.')
    def test_146_oscillating_1(self):
        # Example flight with noisy alt aal
        array = load_compressed(os.path.join(test_data_path, 'find_low_alts_alt_aal_1.npz'))
        alt_aal = P('Altitude AAL For Flight Phases', frequency=2, array=array)
        
        level_flights = buildsections('Level Flight',
            (1856.0, 2392.0),
            (4062.0, 4382.0),
            (4432.0, 4584.0),
            (4606.0, 4856.0),
            (5210.0, 5562.0),
            (5576.0, 5700.0),
            (5840.0, 5994.0),
            (6152.0, 6598.0),
            (7268.0, 7768.0),
            (8908.0, 9124.0),
            (9752.0, 9898.0),
            (9944.0, 10210.0),
            (10814.0, 11098.0),
            (11150.0, 11332.0),
            (11352.0, 11676.0),
            (12122.0, 12346.0),
            (12814.0, 12998.0),
            (13028.0, 13194.0),
            (13432.0, 13560.0),
            (13716.0, 13888.0),
            (13904.0, 14080.0),
            (14122.0, 14348.0),
            (14408.0, 14570.0),
            (14596.0, 14786.0),
            (15092.0, 15356.0),
            (15364.0, 15544.0),
            (15936.0, 16066.0),
            (16078.0, 16250.0),
            (16258.0, 16512.0),
            (16632.0, 16782.0),
            (16854.0, 16982.0),
            (17924.0, 18112.0),
            (18376.0, 18514.0),
            (18654.0, 20582.0),
            (21184.0, 21932.0),
        )
        level_flights.hz = 2
        
        lands = buildsection('Landing', 22296, 22400)
        lands.hz = 2
        
        app_lands = ApproachAndLanding(frequency=2)
        
        app_lands.get_derived([alt_aal, level_flights, lands])
        app_lands = app_lands.get_slices()
        self.assertEqual(len(app_lands), 3)
        self.assertAlmostEqual(app_lands[0].start, 3037, places=0)
        self.assertAlmostEqual(app_lands[0].stop, 4294, places=0)
        self.assertAlmostEqual(app_lands[1].start, 8176, places=0)
        self.assertAlmostEqual(app_lands[1].stop, 8920, places=0)
        self.assertAlmostEqual(app_lands[2].start, 21932, places=0)
        self.assertAlmostEqual(app_lands[2].stop, 22400, places=0)
    
    @unittest.skip('Algorithm is confused by oscillating altitude.')
    def test_146_oscillating_2(self):
        # Example flight with noisy alt aal
        alt_aal = load(os.path.join(test_data_path, 'ApproachAndLanding_alt_aal_1.nod'))
        level_flights = load(os.path.join(test_data_path, 'ApproachAndLanding_level_flights_1.nod'))
        landings = load(os.path.join(test_data_path, 'ApproachAndLanding_landings_1.nod'))
        
        app_lands = ApproachAndLanding(frequency=2)
        
        app_lands.get_derived([alt_aal, level_flights, landings])
        app_lands = app_lands.get_slices()
        self.assertEqual(len(app_lands), 2)
        self.assertAlmostEqual(app_lands[0].start, 8824, places=0)
        self.assertAlmostEqual(app_lands[0].stop, 9980, places=0)
        self.assertAlmostEqual(app_lands[1].start, 29173, places=0)
        self.assertAlmostEqual(app_lands[1].stop, 29726, places=0)


class TestApproach(unittest.TestCase):
    
    def test_can_operate(self):
        node = Approach
        start_stop = Attribute('Segment Type', 'START_AND_STOP')
        ground_only = Attribute('Segment Type', 'GROUND_ONLY')
        self.assertTrue(node.can_operate(('Altitude AAL For Flight Phases', 'Level Flight', 'Landing',),
                                             seg_type=start_stop))
        self.assertTrue(node.can_operate(('Altitude AAL For Flight Phases',),
                                             seg_type=start_stop))
        self.assertTrue(node.can_operate(('Altitude AAL For Flight Phases', 'Landing'),
                                             seg_type=start_stop))
        self.assertTrue(node.can_operate(('Altitude AAL For Flight Phases', 'Level Flight'),
                                             seg_type=start_stop))
        
        self.assertFalse(node.can_operate(('Altitude AAL For Flight Phases', 'Level Flight'),
                                              seg_type=ground_only))

    def test_approach_basic(self):
        alt = np.ma.array(range(5000, 500, -500) + [50] + [0] * 10)
        app = Approach()
        land = buildsection('Landing', 11,14)
        app.derive(Parameter('Altitude AAL For Flight Phases', alt), None, land)
        self.assertEqual(app.get_slices(), [slice(4.0, 9)])

    def test_ignore_takeoff(self):
        alt = np.ma.array([0]*5+range(0,5000,500)+range(5000,500,-500)+[0]*5)
        app = Approach()
        land = buildsection('Landing', 11,14)
        app.derive(Parameter('Altitude AAL For Flight Phases', alt), None, land)
        self.assertEqual(len(app), 1)
        
    def test_with_go_around_and_climbout_atr42_data(self):
        alt_aal = load(os.path.join(test_data_path,
                                    'AltitudeAAL_ATR42_two_goarounds.nod'))
        #alt_aal.array[9860:10180] = 2560
        landing = buildsection('Landing', 27350, 27400)
        level_flights = buildsections('Level Flight',
            (8754, 8971),
            (9844, 10207),
            (11108, 11241),
            (11256, 11489),
            (14887, 20037),
            (20127, 22946),
            (22947, 23587),
            (23701, 24428),
            (25208, 25356),
            (25376, 25837),
        )
        app = Approach()
        app.derive(alt_aal, level_flights, landing)
        
        self.assertEqual(len(app), 3)
        self.assertAlmostEqual(app[0].slice.start,10207, places=0)
        self.assertAlmostEqual(app[0].slice.stop, 10810, places=0)
        self.assertAlmostEqual(app[1].slice.start,12050, places=0)
        self.assertAlmostEqual(app[1].slice.stop, 12631, places=0)
        self.assertAlmostEqual(app[2].slice.start,26926, places=0)
        self.assertAlmostEqual(app[2].slice.stop, 27342, places=0)


class TestBouncedLanding(unittest.TestCase):
    def test_bounce_basic(self):
        fast = buildsection('Fast',2,13)
        airborne = buildsection('Airborne', 3,10)
        alt = np.ma.array([0,0,0,2,10,30,10,2,0,0,0,0,0,0])
        bl = BouncedLanding()
        bl.derive(Parameter('Altitude AAL For Flight Phases', alt), airborne, fast)
        expected = []
        self.assertEqual(bl, expected)

    def test_bounce_with_bounce(self):
        fast = buildsection('Fast',2,13)
        airborne = buildsection('Airborne', 3,7)
        alt = np.ma.array([0,0,0,2,10,30,10,2,0,3,3,0,0,0])
        bl = BouncedLanding()
        bl.derive(Parameter('Altitude AAL For Flight Phases', alt), airborne, fast)
        self.assertEqual(bl[0].slice, slice(9, 11))

    def test_bounce_with_double_bounce(self):
        fast = buildsection('Fast',2,13)
        airborne = buildsection('Airborne', 3,7)
        alt = np.ma.array([0,0,0,2,10,30,10,2,0,3,0,5,0])
        bl = BouncedLanding()
        bl.derive(Parameter('Altitude AAL For Flight Phases', alt), airborne, fast)
        self.assertEqual(bl[0].slice, slice(9, 12))

    def test_bounce_not_detected_with_multiple_touch_and_go(self):
        # test data is a training flight with many touch and go
        bl = BouncedLanding()
        aal = load(os.path.join(test_data_path, 'alt_aal_training.nod'))
        airs = load(os.path.join(test_data_path, 'airborne_training.nod'))
        fast = load(os.path.join(test_data_path, 'fast_training.nod'))
        bl.derive(aal, airs, fast)
        # should not create any bounced landings (used to create 20 at 8000ft)
        self.assertEqual(len(bl), 0)


class TestILSGlideslopeEstablished(unittest.TestCase):
    def test_can_operate(self):
        expected=[('ILS Glideslope', 'ILS Localizer Established',
                   'Altitude AAL For Flight Phases')]
        opts = ILSGlideslopeEstablished.get_operational_combinations()
        self.assertEqual(opts, expected)

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        hdf_copy = copy_file(os.path.join(test_data_path, 'coreg.hdf5'),
                             postfix='_test_copy')
        result = process_flight(hdf_copy, {
            'Engine': {'classification': 'JET',
                       'quantity': 2},
            'Frame': {'doubled': False, 'name': '737-3C'},
            'id': 1,
            'Identifier': '1000',
            'Model': {'family': 'B737 NG',
                      'interpolate_vspeeds': True,
                      'manufacturer': 'Boeing',
                      'model': 'B737-86N',
                      'precise_positioning': True,
                      'series': 'B737-800'},
            'Recorder': {'name': 'SAGEM', 'serial': '123456'},
            'Tail Number': 'G-DEMA'})
        phases = result['phases']
        sections = phases.get(name='ILS Glideslope Established')
        sections
        self.assertTrue(False, msg='Test not implemented.')


class TestILSLocalizerEstablished(unittest.TestCase):
    def test_can_operate(self):
        opts = ILSLocalizerEstablished.get_operational_combinations()
        self.assertIn(('ILS Localizer', 'Altitude AAL For Flight Phases', 'Approach And Landing'), opts)
        self.assertIn(('ILS Localizer', 'Altitude AAL For Flight Phases', 'Approach And Landing', 'ILS Frequency'), opts)
        self.assertIn(('ILS Localizer', 'Altitude AAL For Flight Phases', 'Approach And Landing', 'ILS Frequency', 'Heading'), opts)
        self.assertIn(('ILS Localizer', 'Altitude AAL For Flight Phases', 'Approach And Landing', 'ILS Frequency', 'Heading', 'Heading During Landing'), opts)

    def test_ils_localizer_established_basic(self):
        ils = P('ILS Localizer',np.ma.arange(-3, 0, 0.3))
        alt_aal = P('Alttiude AAL For Flight Phases',
                    np.ma.arange(1000, 0, -100))
        app = S(items=[Section('Approach', slice(0, 10), 0, 10)])
        establish = ILSLocalizerEstablished()
        establish.derive(ils, alt_aal, app, None)
        expected = buildsection('ILS Localizer Established', 7, 10)
        # Slightly daft choice of ils array makes exact equality impossible!
        self.assertAlmostEqual(establish.get_first().start_edge,
                               expected.get_first().start_edge, places=0)

    def test_ils_localizer_no_frequency(self):
        ils = P('ILS Localizer',np.ma.arange(-3, 0, 0.3))
        alt_aal = P('Alttiude AAL For Flight Phases',
                    np.ma.arange(1000, 0, -100))
        app = S(items=[Section('Approach', slice(0, 10), 0, 10)])
        # Tuned to a VOR frequency, the function filter_vor_ils_frequencies
        # will have masked this as not an ILS signal.
        freq = P('ILS Frequency',np.ma.array(data=[108.05]*10,mask=[True]*10))
        establish = ILSLocalizerEstablished()
        establish.derive(ils, alt_aal, app, freq)
        self.assertAlmostEqual(establish.get_first(),None)

    def test_ils_localizer_established_masked_preamble(self):
        '''
        Same data as basic test but has masked ils data before and after
        '''
        ils_array = np.ma.zeros(50)
        ils_array.mask = True
        ils_array[20:30] = np.ma.arange(-3, 0, 0.3)
        ils = P('ILS Localizer', ils_array)
        alt_aal = P('Alttiude AAL For Flight Phases',
                    np.ma.arange(1000, 0, -100))
        app = S(items=[Section('Approach', slice(0, 50), 0, 50)])
        establish = ILSLocalizerEstablished()
        establish.derive(ils, alt_aal, app, None)
        expected = buildsection('ILS Localizer Established', 20+7, 30)
        # Slightly daft choice of ils array makes exact equality impossible!
        self.assertAlmostEqual(establish.get_first().start_edge,
                               expected.get_first().start_edge, places=0)

    def test_ils_localizer_established_never_on_loc(self):
        ils = P('ILS Localizer',np.ma.array([3]*10))
        alt_aal = P('Alttiude AAL For Flight Phases', np.ma.arange(1000, 0,-100))
        app = S(items=[Section('Approach', slice(2, 9), 2, 9)])
        establish = ILSLocalizerEstablished()
        self.assertEqual(establish.derive(ils, alt_aal, app, None), None)

    def test_ils_localizer_established_always_on_loc(self):
        ils = P('ILS Localizer',np.ma.array([-0.2]*10), frequency=0.5)
        alt_aal = P('Altitude AAL For Flight Phases', np.ma.arange(1000, 0,-100))
        app = S(items=[Section('Approach', slice(2, 9), 2, 9)])
        establish = ILSLocalizerEstablished()
        establish.derive(ils, alt_aal, app, None)
        expected = buildsection('ILS Localizer Established',2, 9)
        self.assertEqual(establish.get_slices(), expected.get_slices())

    def test_ils_localizer_established_only_last_segment(self):
        app = S(items=[Section('Approach', slice(2, 9), 2, 9)])
        alt_aal = P('Alttiude AAL For Flight Phases', np.ma.arange(1000, 0,-100))
        ils = P('ILS Localizer',np.ma.array([0,0,0,1,3,3,2,1,0,0]), frequency=0.5)
        establish = ILSLocalizerEstablished()
        establish.derive(ils, alt_aal, app, None)
        expected = buildsection('ILS Localizer Established', 7, 9)
        self.assertEqual(int(establish.get_slices()[0].start), 
                         expected.get_slices()[0].start)
        self.assertEqual(establish.get_slices()[0].stop, 
                         expected.get_slices()[0].stop)


    def test_ils_localizer_stays_established_with_large_visible_deviations(self):
        app = S(items=[Section('Approach', slice(1, 9), 1, 9)])
        alt_aal = P('Alttiude AAL For Flight Phases', np.ma.arange(1000, 0,-100))
        ils = P('ILS Localizer',np.ma.array([0,0,0,1,2.3,2.3,2,1,0,0]))
        establish = ILSLocalizerEstablished()
        establish.derive(ils, alt_aal, app, None)
        expected = buildsection('ILS Localizer Established', 1, 9)
        self.assertEqual(establish.get_slices(), expected.get_slices())

    def test_ils_localizer_insensitive_to_few_masked_values(self):
        app = S(items=[Section('Approach', slice(1, 9), 1, 9)])
        alt_aal = P('Alttiude AAL For Flight Phases', np.ma.arange(1000, 0,-100))
        ils = P('ILS Localizer',np.ma.array(data=[0,0,0,1,2.3,2.3,2,1,0,0],
                                            mask=[0,0,0,0,0,1,1,0,0,0]))
        establish = ILSLocalizerEstablished()
        establish.derive(ils, alt_aal, app, None)
        expected = buildsection('ILS Localizer Established', 1, 9)
        self.assertEqual(establish.get_slices(), expected.get_slices())

    def test_ils_localizer_skips_too_many_masked_values(self):
        app = S(items=[Section('Approach', slice(1, 9), 1, 9)])
        alt_aal = P('Alttiude AAL For Flight Phases', np.ma.arange(1000, 0,-100))
        ils = P('ILS Localizer',np.ma.array(data=[0.0]*20,
                                            mask=[0,1]*10))
        establish = ILSLocalizerEstablished()
        establish.derive(ils, alt_aal, app, None)
        self.assertEqual(establish, [])

    def test_ils_localizer_skips_too_few_values(self):
        app = S(items=[Section('Approach', slice(4, 18), 4, 18)])
        alt_aal = P('Alttiude AAL For Flight Phases', np.ma.arange(1000, 0,-50))
        ils = P('ILS Localizer',np.ma.array(data=[0.0]*20,
                                            mask=[1]*20))
        ils.array.mask[10:15] = 0
        establish = ILSLocalizerEstablished()
        establish.derive(ils, alt_aal, app, None)
        self.assertEqual(establish, [])
 
    def test_ils_localizer_all_masked(self):
        app = S(items=[Section('Approach', slice(2, 9), 2, 9)])
        alt_aal = P('Alttiude AAL For Flight Phases', np.ma.arange(1000, 0,-100))
        ils = P('ILS Localizer',np.ma.array(data=[0.0]*5,
                                            mask=[1]*5))
        establish = ILSLocalizerEstablished()
        establish.derive(ils, alt_aal, app, None)
        self.assertEqual(establish, [])


    def test_ils_localizer_multiple_frequencies(self):
        ils_loc = load(os.path.join(test_data_path, 'ILS_localizer_established_ILS_localizer.nod'))
        ils_freq  = load(os.path.join(test_data_path, 'ILS_localizer_established_ILS_frequency.nod'))
        apps = load(os.path.join(test_data_path, 'ILS_localizer_established_approach.nod'))
        alt_aal = load(os.path.join(test_data_path, 'ILS_localizer_established_alt_aal.nod'))
        establish = ILSLocalizerEstablished()
        establish.derive(ils_loc, alt_aal, apps, ils_freq)
        self.assertEqual(len(establish), 2)
        self.assertAlmostEqual(establish[0].slice.start, 12216, places=0)
        self.assertAlmostEqual(establish[0].slice.stop, 12244, places=0)
        self.assertAlmostEqual(establish[1].slice.start, 12296, places=0)
        self.assertAlmostEqual(establish[1].slice.stop, 12363, places=0)


class TestInitialApproach(unittest.TestCase):
    def test_can_operate(self):
        expected = [('Altitude AAL For Flight Phases',
                     'Approach')]
        opts = InitialApproach.get_operational_combinations()
        self.assertEqual(opts, expected)

    def test_initial_approach_phase_basic(self):
        alt = np.ma.array(range(4000,0,-500)+range(0,4000,500))
        app = InitialApproach()
        alt_aal = Parameter('Altitude AAL For Flight Phases',alt)
        app_land = SectionNode('Approach',
            items=[Section('Approach', slice(2, 8), 2, 8)])
        # Pretend we are flying over flat ground, so the altitudes are equal.
        app.derive(alt_aal, app_land)
        expected = [Section('Initial Approach', slice(2, 6,), 2, 6)]
        self.assertEqual(app, expected)

    def test_initial_approach_phase_over_high_ground(self):
        alt_aal = np.ma.array(range(0,4000,500) + range(4000,0,-500))
        # Raising the ground makes the radio altitude trigger one sample sooner.
        app = InitialApproach()
        alt_aal = Parameter('Altitude AAL For Flight Phases',alt_aal)
        app_land = SectionNode('Approach',
            items=[Section('Approach', slice(10, 16, None), 10, 16)])
        app.derive(alt_aal, app_land)
        expected = [Section('Initial Approach', slice(10, 14), 10, 14)]
        self.assertEqual(app, expected)

    def test_initial_approach_phase_with_go_around(self):
        alt = np.ma.array(range(4000,2000,-500)+range(2000,4000,500))
        app = InitialApproach()
        alt_aal = Parameter('Altitude AAL For Flight Phases',alt)
        app_land = SectionNode('Approach',
            items=[Section('Approach', slice(2, 5), 2, 5)])
        # Pretend we are flying over flat ground, so the altitudes are equal.
        app.derive(alt_aal, app_land)
        expected = [Section('Initial Approach', slice(2, 4), 2, 4)]
        self.assertEqual(app, expected)


'''
class TestCombinedClimb(unittest.TestCase):
    def test_can_operate(self):
        expected = [('Top Of Climb', 'Go Around', 'Liftoff', 'Touchdown')]
        opts = CombinedClimb.get_operational_combinations()
        self.assertEqual(opts, expected)

    def test_derive(self):
        toc_name = 'Top Of Climb'
        toc = KTI(toc_name, items=[KeyTimeInstance(4344, toc_name),
                                   KeyTimeInstance(5496, toc_name),
                                   KeyTimeInstance(7414, toc_name)])
        ga_name = 'Go Around'
        ga = KTI(ga_name, items=[KeyTimeInstance(5404.4375, ga_name),
                                 KeyTimeInstance(6314.9375, ga_name)])
        lo = KTI('Liftoff', items=[KeyTimeInstance(3988.9375, 'Liftoff')])
        node = CombinedClimb()
        node.derive(toc, ga, lo)
        climb_name = 'Combined Climb'
        expected = [
            Section(name='Combined Climb', slice=slice(3988.9375, 4344, None), start_edge=3988.9375, stop_edge=4344),
            Section(name='Combined Climb', slice=slice(5404.4375, 5496, None), start_edge=5404.4375, stop_edge=5496),
            Section(name='Combined Climb', slice=slice(6314.9375, 7414, None), start_edge=6314.9375, stop_edge=7414),
        ]

        self.assertEqual(list(node), expected)
'''


class TestClimbCruiseDescent(unittest.TestCase):
    def test_can_operate(self):
        expected = [('Altitude STD Smoothed','Airborne')]
        opts = ClimbCruiseDescent.get_operational_combinations()
        self.assertEqual(opts, expected)

    def test_climb_cruise_descent_start_midflight(self):
        # This test will find out if we can separate the two humps on this camel
        camel = ClimbCruiseDescent()
        # Needs to get above 15000ft and below 10000ft to create this phase.
        testwave = np.ma.array([15000] * 5 + range(15000, 1000, -1000))
        alt_aal = Parameter('Altitude STD Smoothed',
                            np.ma.array(testwave))
        air=buildsection('Airborne', None, 18)
        camel.derive(alt_aal, air)
        self.assertEqual(camel[0].slice, slice(None, 18))

    def test_climb_cruise_descent_end_midflight(self):
        # This test will find out if we can separate the two humps on this camel
        camel = ClimbCruiseDescent()
        # Needs to get above 15000ft and below 10000ft to create this phase.
        testwave = np.ma.array(range(1000,15000,1000)+[15000]*5)
        alt_aal = Parameter('Altitude STD Smoothed',
                            np.ma.array(testwave))
        air=buildsection('Airborne',0, None)
        camel.derive(alt_aal, air)
        expected = buildsection('Climb Cruise Descent', 0, None)
        self.assertEqual(camel, expected)

    def test_climb_cruise_descent_all_high(self):
        # This test will find out if we can separate the two humps on this camel
        camel = ClimbCruiseDescent()
        # Needs to get above 15000ft and below 10000ft to create this phase.
        testwave = np.ma.array([15000]*5)
        # plot_parameter (testwave)
        air=buildsection('Airborne',0,5)
        camel.derive(Parameter('Altitude STD Smoothed',
                               np.ma.array(testwave)),air)
        expected = []
        self.assertEqual(camel, expected)

    def test_climb_cruise_descent_one_humps(self):
        # This test will find out if we can separate the two humps on this camel
        camel = ClimbCruiseDescent()
        # Needs to get above 15000ft and below 10000ft to create this phase.
        testwave = np.ma.cos(np.arange(0, 3.14 * 2, 0.1)) * (-3000) + 12500
        air=buildsection('Airborne', 0, 62)
        camel.derive(Parameter('Altitude STD Smoothed',
                               np.ma.array(testwave)), air)
        self.assertEqual(len(camel), 1)

    def test_climb_cruise_descent_two_humps(self):
        # This test will find out if we can separate the two humps on this camel
        camel = ClimbCruiseDescent()
        # Needs to get above 15000ft and below 10000ft to create this phase.
        testwave = np.ma.cos(np.arange(0, 3.14 * 4, 0.1)) * (-3000) + 12500
        # plot_parameter (testwave)
        air=buildsection('Airborne',0,122)
        camel.derive(Parameter('Altitude STD Smoothed',
                               np.ma.array(testwave)), air)
        self.assertEqual(len(camel), 2)

    def test_climb_cruise_descent_three_humps(self):
        # This test will find out if we can separate the two humps on this camel
        camel = ClimbCruiseDescent()
        # Needs to get above 15000ft and below 10000ft to create this phase.
        testwave = np.ma.cos(np.arange(0, 3.14 * 6, 0.1)) * (-3000) + 12500
        # plot_parameter (testwave)
        air=buildsection('Airborne',0,186)
        camel.derive(Parameter('Altitude STD Smoothed',
                               np.ma.array(testwave)), air)
        self.assertEqual(len(camel), 3)



'''
# ClimbFromBottomOfDescent is commented out in flight_phase.py
class TestClimbFromBottomOfDescent(unittest.TestCase):
    def test_can_operate(self):
        expected = [('Top Of Climb', 'Climb Start', 'Bottom Of Descent')]
        opts = ClimbFromBottomOfDescent.get_operational_combinations()
        self.assertEqual(opts, expected)

    def test_descent_to_bottom_of_descent_basic(self):
        testwave = np.cos(np.arange(0,12.6,0.1))*(-3000)+12500
        alt_data = np.ma.array(testwave)

        #===========================================================
        # This block of code replicates normal opeartion and ensures
        # that the cruise/climb/descent, top of climb and top of
        # descent data matches the cruise phase under test.
        #===========================================================
        # Use the same test data for flight phases and measured altitude.
        alt = Parameter('Altitude STD', alt_data)

        ccd = ClimbCruiseDescent()
        ccd.derive(Parameter('Altitude For Flight Phases', alt_data))
        toc = TopOfClimb()
        toc.derive(alt, ccd)
        dlc = DescentLowClimb()
        dlc.derive(alt)
        bod = BottomOfDescent()
        bod.derive(alt, dlc)

        descent_phase = ClimbFromBottomOfDescent()
        descent_phase.derive(toc, [], bod) # TODO: include start of climb instance
        expected = [Section(name='Climb From Bottom Of Descent',slice=slice(63, 94, None))]
        self.assertEqual(descent_phase, expected)
'''


class TestClimbing(unittest.TestCase):
    def test_can_operate(self):
        expected = [('Vertical Speed For Flight Phases', 'Airborne')]
        opts = Climbing.get_operational_combinations()
        self.assertEqual(opts, expected)

    def test_climbing_basic(self):
        vert_spd_data = np.ma.array(range(500,1200,100) +
                                    range(1200,-1200,-200) +
                                    range(-1200,500,100))
        vert_spd = Parameter('Vertical Speed For Flight Phases',
                             np.ma.array(vert_spd_data))
        air = buildsection('Airborne', 2, 7)
        up = Climbing()
        up.derive(vert_spd, air)
        self.assertEqual(up[0].slice, slice(3,8))


class TestCruise(unittest.TestCase):
    def test_can_operate(self):
        expected = [('Climb Cruise Descent',
                     'Top Of Climb', 'Top Of Descent')]
        opts = Cruise.get_operational_combinations()
        self.assertEqual(opts, expected)

    def test_cruise_phase_basic(self):
        alt_data = np.ma.array(
            np.cos(np.arange(0, 12.6, 0.1)) * -3000 + 12500)

        #===========================================================
        # This block of code replicates normal opeartion and ensures
        # that the cruise/climb/descent, top of climb and top of
        # descent data matches the cruise phase under test.
        #===========================================================
        # Use the same test data for flight phases and measured altitude.
        alt_p = Parameter('Altitude STD', alt_data)
        # Transform the "recorded" altitude into the CCD input data.
        ccd = ClimbCruiseDescent()
        ccd.derive(alt_p, buildsection('Airborne', 0, len(alt_data)-1))
        toc = TopOfClimb()
        toc.derive(alt_p, ccd)
        tod = TopOfDescent()
        tod.derive(alt_p, ccd)

        test_phase = Cruise()
        test_phase.derive(ccd, toc, tod)
        #===========================================================

        # With this test waveform, the peak at 31:32 is just flat enough
        # for the climb and descent to be a second apart, whereas the peak
        # at 94 genuinely has no interval with a level cruise.
        expected = [slice(31, 32),slice(94, 95)]
        self.assertEqual(test_phase.get_slices(), list(expected))

    def test_cruise_truncated_start(self):
        alt_data = np.ma.array([15000]*5+range(15000,2000,-4000))
        #===========================================================
        alt = Parameter('Altitude STD', alt_data)
        ccd = ClimbCruiseDescent()
        ccd.derive(Parameter('Altitude For Climb Cruise Descent', alt_data),
                   buildsection('Airborne', 0, len(alt_data)))
        toc = TopOfClimb()
        toc.derive(alt, ccd)
        tod = TopOfDescent()
        tod.derive(alt, ccd)
        test_phase = Cruise()
        test_phase.derive(ccd, toc, tod)
        #===========================================================
        self.assertEqual(test_phase[0].slice, slice(None,5))
        self.assertEqual(len(toc), 0)
        self.assertEqual(len(tod), 1)

    def test_cruise_truncated_end(self):
        alt_data = np.ma.array(range(300,36000,6000)+[36000]*4)
        #===========================================================
        alt = Parameter('Altitude STD', alt_data)
        ccd = ClimbCruiseDescent()
        ccd.derive(Parameter('Altitude For Climb Cruise Descent', alt_data),
                   buildsection('Airborne', 0, len(alt_data)))
        toc = TopOfClimb()
        toc.derive(alt, ccd)
        tod = TopOfDescent()
        tod.derive(alt, ccd)
        test_phase = Cruise()
        test_phase.derive(ccd, toc, tod)
        #===========================================================
        expected = Cruise()
        expected.create_section(slice(6, 7), 'Cruise')
        self.assertEqual(test_phase, expected)
        self.assertEqual(len(toc), 1)
        self.assertEqual(len(tod), 0)


class TestInitialClimb(unittest.TestCase):

    def test_can_operate(self):
        expected = [('Takeoff', 'Climb Start', 'Top Of Climb')]
        opts = InitialClimb.get_operational_combinations()
        self.assertEqual(opts, expected)
    
    def test_basic(self):
        ini_clb = InitialClimb()
        toff = buildsection('Takeoff', 10, 20)
        clb_start = build_kti('Climb Start', 30)
        toc = build_kti('Top Of Climb', 40)
        ini_clb.derive(toff, clb_start, toc)
        self.assertEqual(len(ini_clb.get_slices()), 1)
        self.assertEqual(ini_clb.get_first().slice.start, 20)
        self.assertEqual(ini_clb.get_first().slice.stop, 30)

    def test_short_climb(self):
        ini_clb = InitialClimb()
        toff = buildsection('Takeoff', 10, 20)
        clb_start = build_kti('Climb Start', None)
        toc = build_kti('Top Of Climb', 40)
        ini_clb.derive(toff, clb_start, toc)
        self.assertEqual(len(ini_clb.get_slices()), 1)
        self.assertEqual(ini_clb.get_first().slice.start, 20)
        self.assertEqual(ini_clb.get_first().slice.stop, 40)


class TestInitialCruise(unittest.TestCase):
    
    def test_basic(self):
        cruise=buildsection('Cruise', 1000,1500)
        ini_cru = InitialCruise()
        ini_cru.derive(cruise)
        self.assertEqual(ini_cru[0].slice, slice(1300,1330))

    def test_short_cruise(self):
        short_cruise=buildsection('Cruise', 1000,1329)
        ini_cru = InitialCruise()
        ini_cru.derive(short_cruise)
        self.assertEqual(len(ini_cru), 0)

    def test_multiple_cruises(self):
        short_cruise=buildsections('Cruise', [1000,1339], [2000,3000])
        ini_cru = InitialCruise()
        ini_cru.derive(short_cruise)
        self.assertEqual(len(ini_cru), 1)
        self.assertEqual(ini_cru[0].slice, slice(1300,1330))


class TestDescentLowClimb(unittest.TestCase):
    def test_can_operate(self):
        node = DescentLowClimb
        start_stop = Attribute('Segment Type', 'START_AND_STOP')
        ground_only = Attribute('Segment Type', 'GROUND_ONLY')
        self.assertTrue(node.can_operate(('Altitude AAL For Flight Phases',),
                                             seg_type=start_stop))
        self.assertTrue(node.can_operate(('Altitude AAL For Flight Phases', 'Level Flight'),
                                             seg_type=start_stop))
        
        self.assertFalse(node.can_operate(('Altitude AAL For Flight Phases', 'Level Flight'),
                                              seg_type=ground_only))        

    def test_descent_low_climb_basic(self):
        # Wave is 5000ft to 0 ft and back up, with climb of 5000ft.
        testwave = np.cos(np.arange(0, 6.3, 0.1)) * (2500) + 2500
        dsc = testwave - testwave[0]
        dsc[32:] = 0.0
        clb = testwave - min(testwave)
        clb[:31] = 0.0
        alt_aal = Parameter('Altitude AAL For Flight Phases',
                            np.ma.array(testwave))
        #descend = Parameter('Descend For Flight Phases', np.ma.array(dsc))
        #climb = Parameter('Climb For Flight Phases', np.ma.array(clb))
        dlc = DescentLowClimb()
        dlc.derive(alt_aal)
        self.assertEqual(len(dlc), 1)
        self.assertAlmostEqual(dlc[0].slice.start, 14, places=0)
        self.assertAlmostEqual(dlc[0].slice.stop, 38, places=0)

    def test_descent_low_climb_inadequate_climb(self):
        testwave = np.cos(np.arange(0, 6.3, 0.1)) * (240) + 2500 # 480ft climb
        clb = testwave - min(testwave)
        clb[:31] = 0.0
        alt_aal = Parameter('Altitude AAL For Flight Phases',
                            np.ma.array(testwave))
        dlc = DescentLowClimb()
        dlc.derive(alt_aal)
        self.assertEqual(len(dlc), 0)


class TestDescending(unittest.TestCase):
    def test_can_operate(self):
        expected = [('Vertical Speed For Flight Phases', 'Airborne')]
        opts = Descending.get_operational_combinations()
        self.assertEqual(opts, expected)

    def test_descending_basic(self):
        vert_spd = Parameter('Vertical Speed For Flight Phases',
                             np.ma.array([0]*2+[1000]*5+[-500]*12+[0]*2))
        air = buildsection('Airborne',3,20)
        phase = Descending()
        phase.derive(vert_spd, air)
        self.assertEqual(phase[0].slice, slice(7,19))

class TestDescent(unittest.TestCase):
    def test_can_operate(self):
        expected = [('Top Of Descent', 'Bottom Of Descent')]
        opts = Descent.get_operational_combinations()
        self.assertEqual(opts, expected)

    def test_descent_basic(self):
        testwave = np.cos(np.arange(0,12.6,0.1))*(-3000)+12500
        alt_data = np.ma.array(testwave)
        air = buildsection('Airborne', 0, len(testwave))

        #===========================================================
        # This block of code replicates normal opeartion and ensures
        # that the cruise/climb/descent, top of climb and top of
        # descent data matches the cruise phase under test.
        #===========================================================
        # Use the same test data for flight phases and measured altitude.
        alt = Parameter('Altitude STD', alt_data)

        ccd = ClimbCruiseDescent()
        ccd.derive(alt, air)
        tod = TopOfDescent()
        tod.derive(alt, ccd)
        bod = BottomOfDescent()
        bod.derive(ccd)

        descent_phase = Descent()
        descent_phase.derive(tod, bod)
        expected = [Section(name='Descent',slice=slice(32,63,None), start_edge = 32, stop_edge = 63),
                    Section(name='Descent',slice=slice(94,125,None), start_edge = 94, stop_edge = 125)]
        self.assertEqual(descent_phase, expected)

class TestFast(unittest.TestCase):
    def test_can_operate(self):
        self.assertEqual(Fast.get_operational_combinations(),
                         [('Airspeed For Flight Phases',)])

    def test_fast_phase_basic(self):
        slow_and_fast_data = np.ma.array(range(60, 120, 10) + [120] * 300 + \
                                         range(120, 50, -10))
        ias = Parameter('Airspeed For Flight Phases', slow_and_fast_data,1,0)
        phase_fast = Fast()
        phase_fast.derive(ias)
        if AIRSPEED_THRESHOLD == 80:
            expected = buildsection('Fast', 2, 311)
        if AIRSPEED_THRESHOLD == 70:
            expected = buildsection('Fast', 1, 312)
        self.assertEqual(phase_fast.get_slices(), expected.get_slices())

    def test_fast_all_fast(self):
        fast_data = np.ma.array([120] * 10)
        ias = Parameter('Airspeed For Flight Phases', fast_data, 1, 0)
        phase_fast = Fast()
        phase_fast.derive(ias)
        # Q: Should we really create no fast sections?
        expected = buildsection('Fast', 0, 10)
        self.assertEqual(phase_fast.get_slices(), expected.get_slices())

    def test_fast_all_slow(self):
        fast_data = np.ma.array([12] * 10)
        ias = Parameter('Airspeed For Flight Phases', fast_data, 1, 0)
        phase_fast = Fast()
        phase_fast.derive(ias)
        self.assertEqual(phase_fast, [])

    def test_fast_slowing_only(self):
        fast_data = np.ma.arange(110, 60, -10)
        ias = Parameter('Airspeed For Flight Phases', fast_data, 1, 0)
        phase_fast = Fast()
        phase_fast.derive(ias)
        expected = buildsection('Fast', 0, 4)
        self.assertEqual(phase_fast.get_slices(), expected.get_slices())

    def test_fast_speeding_only(self):
        fast_data = np.ma.arange(60, 120, 10)
        ias = Parameter('Airspeed For Flight Phases', fast_data, 1, 0)
        phase_fast = Fast()
        phase_fast.derive(ias)
        expected = buildsection('Fast', 2, 6)
        self.assertEqual(phase_fast.get_slices(), expected.get_slices())
    
    def test_fast_real_data_1(self):
        airspeed = load(os.path.join(test_data_path, 'Fast_airspeed.nod'))
        node = Fast()
        node.derive(airspeed)
        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].slice.start, 2258)
        self.assertEqual(node[0].slice.stop, 14976)
        
        # Test short masked section does not create two fast slices.
        airspeed.array[5000:5029] = np.ma.masked
        node = Fast()
        node.derive(airspeed)
        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].slice.start, 2258)
        self.assertEqual(node[0].slice.stop, 14976)
        
        # Test long masked section splits up fast into two slices.
        airspeed.array[6000:10000] = np.ma.masked
        node = Fast()
        node.derive(airspeed)
        self.assertEqual(len(node), 2)
        self.assertEqual(node[0].slice.start, 2258)
        self.assertEqual(node[0].slice.stop, 6000)
        self.assertEqual(node[1].slice.start, 10000)
        self.assertEqual(node[1].slice.stop, 14976)

    #def test_fast_phase_with_masked_data(self): # These tests were removed.
    #We now use Airspeed For Flight Phases which has a repair mask function,
    #so this is not applicable.

class TestGrounded(unittest.TestCase):
    def test_can_operate(self):
        expected = [
            ('HDF Duration',),
            ('Airborne', 'HDF Duration'),
            ('Airspeed For Flight Phases', 'HDF Duration'),
            ('Airborne', 'Airspeed For Flight Phases', 'HDF Duration')
        ]

        self.assertEqual(Grounded.get_operational_combinations(), expected)

    def test_grounded_phase_basic(self):
        slow_and_fast_data = \
            np.ma.array(range(60, 120, 10) + [120] * 300 + range(120, 50, -10))
        ias = Parameter('Airspeed For Flight Phases', slow_and_fast_data, 1, 0)
        duration = A('HDF Duration', len(ias.array)/ias.frequency)
        air = buildsection('Airborne', 2, 311)
        phase_grounded = Grounded()
        phase_grounded.derive(air, ias, duration)
        expected = buildsections('Grounded', [0, 2], [311, 313])
        self.assertEqual(phase_grounded.get_slices(), expected.get_slices())

    def test_grounded_all_fast(self):
        grounded_data = np.ma.array([120] * 10)
        ias = Parameter('Airspeed For Flight Phases', grounded_data, 1, 0)
        duration = A('HDF Duration', len(ias.array)/ias.frequency)
        air = buildsection('Airborne', None, None)
        phase_grounded = Grounded()
        phase_grounded.derive(air, ias, duration)
        expected = buildsection('Grounded', None, None)
        self.assertEqual(phase_grounded.get_slices(), expected.get_slices())

    def test_grounded_all_slow(self):
        grounded_data = np.ma.array([12]*10)
        ias = Parameter('Airspeed For Flight Phases', grounded_data, 1, 0)
        duration = A('HDF Duration', len(ias.array)/ias.frequency)
        air = buildsection('Airborne', None, None)
        phase_grounded = Grounded()
        phase_grounded.derive(air, ias, duration)
        expected = buildsection('Grounded', 0, 9)
        self.assertEqual(phase_grounded.get_first().slice, expected[0].slice)

    def test_grounded_landing_only(self):
        grounded_data = np.ma.arange(110,60,-10)
        ias = Parameter('Airspeed For Flight Phases', grounded_data,1,0)
        duration = A('HDF Duration', len(ias.array)/ias.frequency)
        air = buildsection('Airborne',None,4)
        phase_grounded = Grounded()
        phase_grounded.derive(air, ias, duration)
        expected = buildsection('Grounded',4,4)
        self.assertEqual(phase_grounded.get_first().slice, expected[0].slice)

    def test_grounded_speeding_only(self):
        grounded_data = np.ma.arange(60,120,10)
        ias = Parameter('Airspeed For Flight Phases', grounded_data,1,0)
        duration = A('HDF Duration', len(ias.array)/ias.frequency)
        air = buildsection('Airborne',2,None)
        phase_grounded = Grounded()
        phase_grounded.derive(air, ias, duration)
        expected = buildsection('Grounded',0,1)
        self.assertEqual(phase_grounded.get_first().slice, expected[0].slice)


class TestFinalApproach(unittest.TestCase):
    def test_can_operate(self):
        expected = [('Altitude AAL For Flight Phases',)]
        opts = FinalApproach.get_operational_combinations()
        self.assertEqual(opts, expected)

    def test_approach_phase_basic(self):
        alt = np.ma.array(range(0,1200,100)+range(1500,500,-100)+range(400,0,-40)+[0,0,0])
        alt_aal = Parameter('Altitude AAL For Flight Phases', array=alt)
        expected = buildsection('Final Approach', 18, 31)
        fapp=FinalApproach()
        fapp.derive(alt_aal)
        self.assertEqual(fapp.get_slices(), expected.get_slices())


class TestGearRetracting(unittest.TestCase):
    def setUp(self):
        self.node_class = GearRetracting
        
    def test_can_operate(self):
        opts = self.node_class.get_operational_combinations()
        self.assertEqual(opts,[
                         ('Gear Up Selected', 'Gear Up', 'Airborne')])

    def test_derive(self):
        gear_down = M('Gear Up', np.ma.array([0,0,0,0,1,1,1,1,1,0,0,0]),
                      values_mapping={0:'Down', 1:'Up'})
        gear_up_sel = M('Gear Up Selected',
                        np.ma.array([0,0,1,1,1,1,1,0,0,0,0,0]),
                        values_mapping={0:'Down', 1:'Up'})
        airs = buildsection('Airborne', 1, 11)
        
        
        node = self.node_class()
        node.derive(gear_up_sel, gear_down, airs)
        expected=buildsection('Gear Retracting', 2, 4)
        self.assertEqual(node.get_slices(), expected.get_slices())


class TestGoAroundAndClimbout(unittest.TestCase):

    def test_can_operate(self):
        node = GoAroundAndClimbout
        start_stop = Attribute('Segment Type', 'START_AND_STOP')
        ground_only = Attribute('Segment Type', 'GROUND_ONLY')
        self.assertTrue(node.can_operate(('Altitude AAL For Flight Phases','Level Flight'),
                                             seg_type=start_stop))
        self.assertTrue(node.can_operate(('Altitude AAL For Flight Phases',),
                                             seg_type=start_stop))
        self.assertFalse(node.can_operate(('Altitude AAL For Flight Phases',),
                                              seg_type=ground_only))

    def test_go_around_and_climbout_basic(self):
        # The Go-Around phase starts 500ft before the minimum altitude is
        # reached, and ends after 2000ft climb....
        height = np.ma.array(range(0,40)+
                             range(40,20,-1)+
                             range(20,45)+
                             [45]*5+
                             range(45,0,-1))*100.0
        alt = Parameter('Altitude For Flight Phases', height)
        levels = buildsection('Level Flight', 85, 91)
        ga_phase = GoAroundAndClimbout()
        ga_phase.derive(alt, levels)
        expected = buildsection('Go Around And Climbout', 55, 80)
        self.assertEqual(len(ga_phase), 1)
        self.assertEqual(ga_phase.get_first().start_edge, expected[0].start_edge)
        self.assertEqual(ga_phase.get_first().stop_edge, expected[0].stop_edge)

    def test_go_around_and_climbout_level_off(self):
        # The Go-Around phase starts 500ft before the minimum altitude is
        # reached, and ends ... or until level-off.
        height = np.ma.array(range(0,40)+
                             range(40,20,-1)+
                             range(20,30)+
                             [30]*5+
                             range(30,0,-1))*100.0
        alt = Parameter('Altitude For Flight Phases', height)
        levels = buildsection('Level Flight', 70, 76)
        ga_phase = GoAroundAndClimbout()
        ga_phase.derive(alt, levels)
        # Level flight reached at ~70
        expected = buildsection('Go Around And Climbout', 55, 68)
        self.assertEqual(len(ga_phase), 1)
        self.assertEqual(ga_phase.get_first().start_edge, expected[0].start_edge)
        self.assertEqual(ga_phase.get_first().stop_edge, expected[0].stop_edge)

    def test_go_around_and_climbout_phase_not_reaching_2000ft(self):
        '''
        down = np.ma.array(range(4000,1000,-490)+[1000]*7) - 4000
        up = np.ma.array([1000]*7+range(1000,4500,490)) - 1500
        ga_kti = KTI('Go Around', items=[KeyTimeInstance(index=7, name='Go Around')])
        ga_phase = GoAroundAndClimbout()
        ga_phase.derive(Parameter('Descend For Flight Phases',down),
                        Parameter('Climb For Flight Phases',up),
                        ga_kti)
        expected = buildsection('Go Around And Climbout', 4.9795918367346941,
                                12.102040816326531)
        self.assertEqual(len(ga_phase), 1)
        self.assertEqual(ga_phase.get_first().start_edge, expected[0].start_edge)
        self.assertEqual(ga_phase.get_first().stop_edge, expected[0].stop_edge)
        '''
        alt_aal = load(os.path.join(test_data_path, 'alt_aal_goaround.nod'))
        level_flights = SectionNode('Level Flight')
        level_flights.create_sections([
            slice(1629.0, 2299.0, None),
            slice(3722.0, 4708.0, None),
            slice(4726.0, 4807.0, None),
            slice(5009.0, 5071.0, None),
            slice(5168.0, 6883.0, None),
            slice(8433.0, 9058.0, None)])
        ga_phase = GoAroundAndClimbout()
        ga_phase.derive(alt_aal, level_flights)
        self.assertEqual(len(ga_phase), 3)
        self.assertAlmostEqual(ga_phase[0].slice.start, 3586, places=0)
        self.assertAlmostEqual(ga_phase[0].slice.stop, 3722, places=0)
        self.assertAlmostEqual(ga_phase[1].slice.start, 4895, places=0)
        self.assertAlmostEqual(ga_phase[1].slice.stop, 5009, places=0)
        self.assertAlmostEqual(ga_phase[2].slice.start, 7124, places=0)
        self.assertAlmostEqual(ga_phase[2].slice.stop, 7265, places=0)
    
    def test_go_around_and_climbout_below_3000ft(self):
        alt_aal = load(os.path.join(test_data_path,
                                    'GoAroundAndClimbout_AltitudeAAL.nod'))
        level_flights = load(os.path.join(test_data_path,
                                          'GoAroundAndClimbout_LevelFlights.nod'))
        ga_phase = GoAroundAndClimbout()
        ga_phase.derive(alt_aal, level_flights)
        self.assertEqual(len(ga_phase), 1)
        self.assertAlmostEqual(ga_phase[0].slice.start, 10697, places=0)
        self.assertAlmostEqual(ga_phase[0].slice.stop, 10968, places=0)

    def test_go_around_and_climbout_real_data(self):
        alt_aal = load(os.path.join(test_data_path,
                                    'GoAroundAndClimbout_alt_aal.nod'))
        
        ga_phase = GoAroundAndClimbout()
        ga_phase.derive(alt_aal, None)
        self.assertEqual(len(ga_phase), 5)
        #self.assertEqual(ga_phase.get_slices(), [
            #slice(1005.0, 1170.0, None),
            #slice(1378.0, 1502.0, None),
            #slice(1676.0, 1836.0, None),
            #slice(2021.0, 2206.0, None),
            #slice(2208.0, 2502.0, None)])
        self.assertAlmostEqual(ga_phase[0].slice.start, 1057, places=0)
        self.assertAlmostEqual(ga_phase[0].slice.stop, 1169, places=0)
        self.assertAlmostEqual(ga_phase[1].slice.start, 1393, places=0)
        self.assertAlmostEqual(ga_phase[1].slice.stop, 1502, places=0)
        # For some reason, places=0 breaks for 1721.5 == 1721 or 1722..
        self.assertAlmostEqual(ga_phase[2].slice.start, 1722, delta=0.5)
        self.assertAlmostEqual(ga_phase[2].slice.stop, 1836, places=0)
        self.assertAlmostEqual(ga_phase[3].slice.start, 2071, places=0)
        self.assertAlmostEqual(ga_phase[3].slice.stop, 2205, places=0)
        self.assertAlmostEqual(ga_phase[4].slice.start, 2392, places=0)
        self.assertAlmostEqual(ga_phase[4].slice.stop, 2502, places=0)
        
    def test_two_go_arounds_for_atr42(self):
        alt_aal = load(os.path.join(test_data_path,
                                    'AltitudeAAL_ATR42_two_goarounds.nod'))
        ga_phase = GoAroundAndClimbout()
        ga_phase.derive(alt_aal, None)
        self.assertEqual(len(ga_phase), 2)
        self.assertAlmostEqual(ga_phase[0].slice.start, 10703, places=0)
        self.assertAlmostEqual(ga_phase[0].slice.stop, 10948, places=0)
        self.assertAlmostEqual(ga_phase[1].slice.start, 12529, places=0)
        self.assertAlmostEqual(ga_phase[1].slice.stop, 12749, places=0)



class TestHolding(unittest.TestCase):
    def test_can_operate(self):
        self.assertEqual(Holding.get_operational_combinations(),
                         [('Altitude AAL For Flight Phases',
                          'Heading Increasing','Latitude Smoothed',
                          'Longitude Smoothed')])

    def test_straightish_not_detected(self):
        hdg=P('Heading Increasing', np.ma.arange(3000)*0.45)
        alt=P('Altitude AAL For Flight Phases', np.ma.array([10000]*3000))
        lat=P('Latitude Smoothed', np.ma.array([24.0]*3000))
        lon=P('Longitude Smoothed', np.ma.array([24.0]*3000))
        hold=Holding()
        hold.derive(alt, hdg, lat, lon)
        expected=[]
        self.assertEqual(hold, expected)

    def test_bent_detected(self):
        hdg=P('Heading Increasing', np.ma.arange(3000)*(1.1))
        alt=P('Altitude AAL For Flight Phases', np.ma.array([25000]+[10000]*3000+[0]))
        lat=P('Latitude Smoothed', np.ma.array([24.0]*3000))
        lon=P('Longitude Smoothed', np.ma.array([24.0]*3000))
        hold=Holding()
        hold.derive(alt, hdg, lat, lon)
        expected=buildsection('Holding',1,3000)
        self.assertEqual(hold.get_slices(), expected.get_slices())

    def test_rejected_outside_height_range(self):
        hdg=P('Heading Increasing', np.ma.arange(3000)*(1.1))
        alt=P('Altitude AAL For Flight Phases', np.ma.arange(3000,0,-1)*10)
        lat=P('Latitude Smoothed', np.ma.array([24.0]*3000))
        lon=P('Longitude Smoothed', np.ma.array([24.0]*3000))
        hold=Holding()
        hold.derive(alt, hdg, lat, lon)
        # OK - I cheated. Who cares about the odd one sample passing 5000ft :o)
        expected=buildsection('Holding', 1001, 2700)
        self.assertEqual(hold.get_slices(), expected.get_slices())

    def test_hold_detected(self):
        rot=[0]*600+([3]*60+[0]*60)*6+[0]*180+([3]*60+[0]*90)*6+[0]*600
        hdg=P('Heading Increasing', integrate(rot,1.0))
        alt=P('Altitude AAL For Flight Phases', np.ma.array([25000]+[10000]*3000)+[0])
        lat=P('Latitude Smoothed', np.ma.array([24.0]*3000))
        lon=P('Longitude Smoothed', np.ma.array([24.0]*3000))
        hold=Holding()
        hold.derive(alt, hdg, lat, lon)
        # Stopped using buildsections because someone changed the endpoints (grrr) DJ.
        # expected=buildsections('Holding',[540,1320],[1440,2370])
        self.assertEqual(hold[0].slice, slice(540, 1320))
        self.assertEqual(hold[1].slice, slice(1440, 2370))

    def test_hold_rejected_if_travelling(self):
        rot=[0]*600+([3]*60+[0]*60)*6+[0]*180+([3]*60+[0]*90)*6+[0]*600
        hdg=P('Heading Increasing', integrate(rot,1.0))
        alt=P('Altitude AAL For Flight Phases', np.ma.array([10000]*3000))
        lat=P('Latitude Smoothed', np.ma.arange(0, 30, 0.01))
        lon=P('Longitude Smoothed', np.ma.arange(0, 30, 0.01))
        hold=Holding()
        hold.derive(alt, hdg, lat, lon)
        expected=[]
        self.assertEqual(hold, expected)

    def test_single_turn_rejected(self):
        rot=np.ma.array([0]*500+[3]*60+[0]*600+[6]*60+[0]*600+[0]*600+[3]*90+[0]*490, dtype=float)
        hdg=P('Heading Increasing', integrate(rot,1.0))
        alt=P('Altitude AAL For Flight Phases', np.ma.array([10000]*3000))
        lat=P('Latitude Smoothed', np.ma.array([24.0]*3000))
        lon=P('Longitude Smoothed', np.ma.array([24.0]*3000))
        hold=Holding()
        hold.derive(alt, hdg, lat, lon)
        expected=[]
        self.assertEqual(hold, expected)



class TestLanding(unittest.TestCase):
    def test_can_operate(self):
        node = Landing
        start_stop = Attribute('Segment Type', 'START_AND_STOP')
        ground_only = Attribute('Segment Type', 'GROUND_ONLY')
        self.assertTrue(node.can_operate(('Heading Continuous', 'Altitude AAL For Flight Phases', 'Fast',),
                                             seg_type=start_stop))
        self.assertTrue(node.can_operate(('Heading Continuous', 'Altitude AAL For Flight Phases',),
                                             seg_type=start_stop))
        self.assertTrue(node.can_operate(('Altitude AAL For Flight Phases', 'Fast'),
                                             seg_type=start_stop))
        self.assertTrue(node.can_operate(('Altitude AAL For Flight Phases'),
                                             seg_type=start_stop))
        
        self.assertFalse(node.can_operate(('Altitude AAL For Flight Phases'),
                                              seg_type=ground_only))

    def test_landing_basic(self):
        head = np.ma.array([20]*8+[10,0])
        alt_aal = np.ma.array([80,40,20,5]+[0]*6)
        phase_fast = buildsection('Fast',0,5)
        landing = Landing()
        landing.derive(P('Heading Continuous',head),
                       P('Altitude AAL For Flight Phases',alt_aal),
                       phase_fast)
        expected = buildsection('Landing', 0.75, 9)
        self.assertEqual(landing.get_slices(), expected.get_slices())

    def test_landing_turnoff(self):
        head = np.ma.array([20]*15+range(20,0,-2))
        alt_aal = np.ma.array([80,40,20,5]+[0]*26)
        phase_fast = buildsection('Fast',0,5)
        landing = Landing()
        landing.derive(P('Heading Continuous',head),
                       P('Altitude AAL For Flight Phases',alt_aal),
                       phase_fast)
        expected = buildsection('Landing', 0.75, 24)
        self.assertEqual(landing.get_slices(), expected.get_slices())

    def test_landing_turnoff_left(self):
        head = np.ma.array([20]*15+range(20,0,-2))*-1.0
        alt_aal = np.ma.array([80,40,20,5]+[0]*26)
        phase_fast = buildsection('Fast', 0, 5)
        landing = Landing()
        landing.derive(P('Heading Continuous',head),
                       P('Altitude AAL For Flight Phases',alt_aal),
                       phase_fast)
        expected = buildsection('Landing', 0.75, 24)
        self.assertEqual(landing.get_slices(), expected.get_slices())
        
    def test_landing_with_multiple_fast(self):
        # ensure that the result is a single phase!
        head = np.ma.array([20]*15+range(20,0,-2))
        alt_aal = np.ma.array(range(140,0,-10)+[0]*26)
        # first test the first section that is not within the landing heights
        phase_fast = buildsections('Fast', [2, 5], [7, 10])
        landing = Landing()
        landing.derive(P('Heading Continuous',head),
                       P('Altitude AAL For Flight Phases',alt_aal),
                       phase_fast)
        self.assertEqual(len(landing), 1)
        self.assertEqual(landing[0].slice.start, 9)
        self.assertEqual(landing[0].slice.stop, 24)

        # second, test both sections are within the landing section of data
        phase_fast = buildsections('Fast', [0, 12], [14, 15])
        landing = Landing()
        landing.derive(P('Heading Continuous',head),
                       P('Altitude AAL For Flight Phases',alt_aal),
                       phase_fast)
        self.assertEqual(len(landing), 1)
        self.assertEqual(landing[0].slice.start, 9)
        self.assertEqual(landing[0].slice.stop, 24)


class TestLandingRoll(unittest.TestCase):
    def test_can_operate(self):
        self.assertEqual(LandingRoll.get_operational_combinations(),
                         [('Groundspeed', 'Landing'),
                          ('Airspeed True', 'Landing'),
                          ('Pitch', 'Groundspeed', 'Landing'),
                          ('Pitch', 'Airspeed True', 'Landing'),
                          ('Groundspeed', 'Airspeed True', 'Landing'),
                          ('Pitch', 'Groundspeed', 'Airspeed True', 'Landing')])


class TestMobile(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = Mobile
        self.operational_combinations = [('Heading Rate',),]

    def test_rot_only(self):
        rot = np.ma.array([0,0,5,5,5,0,0])
        move = Mobile()
        move.derive(P('Heading Rate',rot))
        expected = buildsection('Mobile', 2, 4)
        self.assertEqual(move.get_slices(), expected.get_slices())


class TestLevelFlight(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = LevelFlight
        self.operational_combinations = [('Airborne', 'Vertical Speed For Flight Phases')]

    def test_level_flight_phase_basic(self):
        data = range(0, 400, 1) + range(400, -450, -1) + range(-450, 50, 1)
        vrt_spd = Parameter(
            name='Vertical Speed For Flight Phases',
            array=np.ma.array(data),
        )
        airborne = SectionNode('Airborne', items=[
            Section('Airborne', slice(0, 3600, None), 0, 3600),
        ])
        level = LevelFlight()
        level.derive(airborne, vrt_spd)
        self.assertEqual(level, [
            Section('Level Flight', slice(0, 301, None), 0, 301), 
            Section('Level Flight', slice(500, 1101, None), 500, 1101), 
            Section('Level Flight', slice(1400, 1750, None), 1400, 1750)])

    def test_level_flight_phase_not_airborne_basic(self):
        data = range(0, 400, 1) + range(400, -450, -1) + range(-450, 50, 1)
        vrt_spd = Parameter(
            name='Vertical Speed For Flight Phases',
            array=np.ma.array(data),
        )
        airborne = SectionNode('Airborne', items=[
            Section('Airborne', slice(550, 1200, None), 550, 1200),
        ])
        level = LevelFlight()
        level.derive(airborne, vrt_spd)
        self.assertEqual(level, [
            Section('Level Flight', slice(550, 1101, None), 550, 1101)
        ])

    def test_rejects_short_segments(self):
        data = [400]*50+[0]*20+[400]*50+[0]*80+[-400]*40+[4]*40+[500]*40
        vrt_spd = Parameter(
            name='Vertical Speed For Flight Phases',
            array=np.ma.array(data),
            frequency=1.0
        )
        airborne = SectionNode('Airborne', items=[
            Section('Airborne', slice(0, 320), 0, 320),
        ])
        level = LevelFlight()
        level.derive(airborne, vrt_spd)
        self.assertEqual(level, [
            Section('Level Flight', slice(120, 200, None), 120, 200)
        ])


class TestStraightAndLevel(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = StraightAndLevel
        self.operational_combinations = [('Level Flight', 'Heading')]

    def test_straight_and_level_basic(self):
        data = [0]*60+range(0, 360, 1) + range(360,0,-2) + [0]*120
        hdg = Parameter('Heading',array=np.ma.array(data))
        level = buildsection('Level Flight', 0, 900)
        s_and_l = StraightAndLevel()
        s_and_l.derive(level, hdg)
        self.assertEqual(s_and_l[1].slice, slice(600, 720, None))
        self.assertEqual(s_and_l[0].slice, slice(0, 426, None))
                         

class TestRejectedTakeoff(unittest.TestCase):
    '''
    The test was originally written for an acceleration threshold of 0.1g.
    This was later increased to 0.15g and the test amended to match that
    scaling, hence the *1.5 factors.
    '''
    def test_can_operate(self):
        expected = [('Acceleration Longitudinal Offset Removed', 'Grounded')]
        self.assertEqual(
            expected,
            RejectedTakeoff.get_operational_combinations())

    def test_derive_basic(self):
        accel_lon = P('Acceleration Longitudinal Offset Removed',
                      np.ma.array([0] * 3 + [0.02, 0.05, 0.11, 0, -0.17,] + [0] * 7 +
                                  [0.2, 0.4, 0.1] + [0.11] * 4 + [0] * 6 + [-2] +
                                  [0] * 5 + [0.02, 0.08, 0.08, 0.08, 0.08] + [0] * 20)*1.5)
        grounded = buildsection('Grounded', 0, len(accel_lon.array))
        
        node = RejectedTakeoff()
        # Set a low frequency to pass slice duration checks.
        node.frequency = 1/64.0
        node.derive(accel_lon, grounded)
        self.assertEqual(len(node), 1)
        self.assertAlmostEqual(node[0].slice.start, 15, 0)
        self.assertAlmostEqual(node[0].slice.stop, 21, 0)
    
    def test_derive_flight_with_rejected_takeoff_1(self):
        accel_lon = load(os.path.join(
            test_data_path,
            'RejectedTakeoff_AccelerationLongitudinalOffsetRemoved_2.nod'))
        accel_lon.array *= 1.5
        grounded = load(os.path.join(test_data_path,
                                     'RejectedTakeoff_Grounded_2.nod'))
        node = RejectedTakeoff()
        node.derive(accel_lon, grounded)
        self.assertEqual(len(node), 1)
        self.assertAlmostEqual(node[0].slice.start, 3612, 0)
        self.assertAlmostEqual(node[0].slice.stop, 3682, 0)
    
    def test_derive_flight_with_rejected_takeoff_2(self):
        accel_lon = load(os.path.join(
            test_data_path,
            'RejectedTakeoff_AccelerationLongitudinalOffsetRemoved_5.nod'))
        accel_lon.array *= 1.5
        grounded = load(os.path.join(test_data_path,
                                     'RejectedTakeoff_Grounded_5.nod'))
        node = RejectedTakeoff()
        node.derive(accel_lon, grounded)
        self.assertEqual(len(node), 1)
        self.assertAlmostEqual(node[0].slice.start, 2373, 0)
        self.assertAlmostEqual(node[0].slice.stop, 2435, 0)
    
    def test_derive_flight_with_rejected_takeoff_short(self):
        '''
        test derived from genuine low speed rejected takeoff FDS hash 452728ea2768
        '''
        accel_lon = load(os.path.join(
            test_data_path,
            'RejectedTakeoff_AccelerationLongitudinalOffsetRemoved_Short.nod'))
        grounded = buildsections('Grounded', [0, 3796], [23516, 24576])
        node = RejectedTakeoff(frequency=4)
        node.derive(accel_lon, grounded)
        self.assertEqual(len(node), 1)
        self.assertAlmostEqual(node[0].slice.start, 1878, 0)
        self.assertAlmostEqual(node[0].slice.stop, 2056, 0)

    def test_derive_flight_without_rejected_takeoff_3(self):
        accel_lon = load(os.path.join(
            test_data_path,
            'RejectedTakeoff_AccelerationLongitudinalOffsetRemoved_4.nod'))
        grounded = load(os.path.join(test_data_path,
                                     'RejectedTakeoff_Grounded_4.nod'))
        accel_lon.array *= 1.5
        node = RejectedTakeoff()
        node.derive(accel_lon, grounded)
        self.assertEqual(len(node), 0)

    def test_derive_flight_without_rejected_takeoff_1(self):
        accel_lon = load(os.path.join(
            test_data_path,
            'RejectedTakeoff_AccelerationLongitudinalOffsetRemoved_1.nod'))
        grounded = load(os.path.join(test_data_path,
                                     'RejectedTakeoff_Grounded_1.nod'))
        accel_lon.array *= 1.5
        node = RejectedTakeoff()
        node.derive(accel_lon, grounded)
        self.assertEqual(len(node), 0)
    
    def test_derive_flight_without_rejected_takeoff_2(self):
        accel_lon = load(os.path.join(
            test_data_path,
            'RejectedTakeoff_AccelerationLongitudinalOffsetRemoved_3.nod'))
        accel_lon.array *= 1.5
        grounded = load(os.path.join(test_data_path,
                                     'RejectedTakeoff_Grounded_3.nod'))
        node = RejectedTakeoff()
        node.derive(accel_lon, grounded)
        self.assertEqual(len(node), 0)


class TestTakeoff(unittest.TestCase):
    def test_can_operate(self):
        # Airborne dependency added to avoid trying to derive takeoff when
        # aircraft never airborne
        self.assertEqual(Takeoff.get_operational_combinations(),
                         [('Heading Continuous', 'Altitude AAL For Flight Phases', 'Fast', 'Airborne')])

    def test_takeoff_basic(self):
        head = np.ma.array([ 0,0,10,20,20,20,20,20,20,20,20])
        alt_aal = np.ma.array([0,0,0,0,0,0,0,0,10,30,70])
        phase_fast = buildsection('Fast', 6.5, 10)
        takeoff = Takeoff()
        takeoff.derive(P('Heading Continuous', head),
                       P('Altitude AAL For Flight Phases', alt_aal),
                       phase_fast)
        expected = buildsection('Takeoff', 1.5, 9.125)
        self.assertEqual(takeoff.get_slices(), expected.get_slices())

    def test_takeoff_with_zero_slices(self):
        '''
        A zero slice was causing the derive method to raise an exception.
        This test aims to replicate the problem, and shows that with a None
        slice an empty takeoff phase is produced.
        '''
        head = np.ma.array([ 0,0,10,20,20,20,20,20,20,20,20])
        alt_aal = np.ma.array([0,0,0,0,0,0,0,0,10,30,70])
        phase_fast = buildsection('Fast', None, None)
        takeoff = Takeoff()
        takeoff.derive(P('Heading Continuous',head),
                       P('Altitude AAL For Flight Phases',alt_aal),
                       phase_fast)
        expected = []
        self.assertEqual(takeoff, expected)


class TestTaxiOut(unittest.TestCase):
    def test_can_operate(self):
        expected = [('Mobile', 'Takeoff', 'First Eng Start Before Liftoff')]
        self.assertEqual(TaxiOut.get_operational_combinations(), expected)

    def test_taxi_out(self):
        gnd = buildsections('Mobile', [0, 1], [3, 9])
        toff = buildsection('Takeoff', 8, 11)
        first_eng_starts = KTI(
            'First Eng Start Before Liftoff',
            items=[KeyTimeInstance(2, 'First Eng Start Before Liftoff')])
        tout = TaxiOut()
        tout.derive(gnd, toff, first_eng_starts)
        expected = buildsection('Taxi Out', 4, 7)
        self.assertEqual(tout.get_slices(), expected.get_slices())
        first_eng_starts = KTI(
            'First Eng Start Before Liftoff',
            items=[KeyTimeInstance(5, 'First Eng Start Before Liftoff')])
        tout = TaxiOut()
        tout.derive(gnd, toff, first_eng_starts)
        expected = buildsection('Taxi Out', 5, 7)
        self.assertEqual(tout.get_slices(), expected.get_slices())


class TestTaxiIn(unittest.TestCase):
    def test_can_operate(self):
        expected = [('Mobile', 'Landing', 'Last Eng Stop After Touchdown')]
        self.assertEqual(TaxiIn.get_operational_combinations(), expected)

    def test_taxi_in_1(self):
        gnd = buildsections('Mobile', [7, 13], [16, 17])
        landing = buildsection('Landing', 5, 11)
        last_eng_stops = KTI(
            'Last Eng Stop After Touchdown',
            items=[KeyTimeInstance(15, 'Last Eng Stop After Touchdown')])
        t_in = TaxiIn()
        t_in.derive(gnd, landing, last_eng_stops)
        expected = buildsection('Taxi In', 12, 14)
        self.assertEqual(t_in.get_slices(),expected.get_slices())
    
    def test_taxi_in_2(self):
        gnd = buildsection('Mobile', 488, 7007)
        landing = buildsection('Landing', 3389, 3437)
        last_eng_stops = KTI(
            'Last Eng Stop After Touchdown',
            items=[KeyTimeInstance(3734, 'Last Eng Stop After Touchdown')])
        t_in = TaxiIn()
        t_in.derive(gnd, landing, last_eng_stops)
        self.assertEqual(len(t_in), 1)
        self.assertEqual(t_in[0].slice.start, 3438)
        self.assertEqual(t_in[0].slice.stop, 3734)


class TestTaxiing(unittest.TestCase):
    def test_can_operate(self):
        combinations = Taxiing.get_operational_combinations()
        self.assertTrue(('Mobile', 'Airborne') in combinations)
        self.assertTrue(('Mobile', 'Takeoff', 'Landing') in combinations)
        self.assertTrue(('Mobile', 'Groundspeed', 'Airborne') in combinations)
        self.assertTrue(
            ('Mobile', 'Groundspeed', 'Takeoff', 'Landing', 'Airborne')
            in combinations)

    def test_taxiing_mobile_airborne(self):
        mobiles = buildsection('Mobile', 10, 90)
        airs = buildsection('Airborne', 20, 80)
        node = Taxiing()
        node.derive(mobiles, None, None, None, None, airs)
        self.assertEqual(len(node), 2)
        self.assertEqual(node[0].slice.start, 10)
        self.assertEqual(node[0].slice.stop, 20)
        self.assertEqual(node[1].slice.start, 80)
        self.assertEqual(node[1].slice.stop, 90)
    
    def test_taxiing_mobile_takeoff_landing(self):
        mobiles = buildsection('Mobile', 10, 90)
        toffs = buildsection('Takeoff', 20, 30)
        lands = buildsection('Landing', 70, 80)
        airs = buildsection('Airborne', 25, 75)
        node = Taxiing()
        node.derive(mobiles, None, toffs, lands, None, airs)
        self.assertEqual(len(node), 2)
        self.assertEqual(node[0].slice.start, 10)
        self.assertEqual(node[0].slice.stop, 20)
        self.assertEqual(node[1].slice.start, 80)
        self.assertEqual(node[1].slice.stop, 90)
    
    def test_taxiing_mobile_takeoff_landing_gspd(self):
        mobiles = buildsection('Mobile', 10, 100)
        gspd_array = np.ma.array([0] * 15 +
                                 [10] * 70 +
                                 [0] * 5 +
                                 [10] * 5 +
                                 [0] * 5)
        gspd = P('Groundspeed', array=gspd_array)
        toffs = buildsection('Takeoff', 20, 30)
        lands = buildsection('Landing', 60, 70)
        airs = buildsection('Airborne', 25, 65)
        node = Taxiing()
        node.derive(mobiles, gspd, toffs, lands, None, airs)
        self.assertEqual(len(node), 3)
        self.assertEqual(node[0].slice.start, 15)
        self.assertEqual(node[0].slice.stop, 20)
        self.assertEqual(node[1].slice.start, 70)
        self.assertEqual(node[1].slice.stop, 85)
        self.assertEqual(node[2].slice.start, 90)
        self.assertEqual(node[2].slice.stop, 95)

    
    def test_taxiing_including_rejected_takeoff(self):
        mobiles = buildsection('Mobile', 3, 100)
        gspd_array = np.ma.array([0] * 10 +
                                 [10] * 30 +
                                 [0] * 5 +
                                 [10] * 50 +
                                 [0] * 5)
        gspd = P('Groundspeed', array=gspd_array)
        toffs = buildsection('Takeoff',  50, 60)
        lands = buildsection('Landing', 80, 90)
        airs = buildsection('Airborne', 55, 85)
        rtos = buildsection('Rejected Takeoff', 20, 30)
        node = Taxiing()
        node.derive(mobiles, gspd, toffs, lands, rtos, airs)
        self.assertEqual(len(node), 4)
        self.assertEqual(node[0].slice.start, 10)
        self.assertEqual(node[0].slice.stop, 20)
        self.assertEqual(node[1].slice.start, 30)
        self.assertEqual(node[1].slice.stop, 40)
        self.assertEqual(node[2].slice.start, 45)
        self.assertEqual(node[2].slice.stop, 50)
        self.assertEqual(node[3].slice.start, 90)
        self.assertEqual(node[3].slice.stop, 95)


class TestTurningInAir(unittest.TestCase):
    def test_can_operate(self):
        expected = [('Heading Rate', 'Airborne')]
        self.assertEqual(TurningInAir.get_operational_combinations(), expected)

    def test_turning_in_air_phase_basic(self):
        rate_of_turn_data = np.arange(-4, 4.4, 0.4)
        rate_of_turn = Parameter('Heading Rate', np.ma.array(rate_of_turn_data))
        airborne = buildsection('Airborne',0,21)
        turning_in_air = TurningInAir()
        turning_in_air.derive(rate_of_turn, airborne)
        expected = buildsections('Turning In Air',[0, 6],[16,21])
        self.assertEqual(turning_in_air.get_slices(), expected.get_slices())

    def test_turning_in_air_phase_with_mask(self):
        rate_of_turn_data = np.ma.arange(-4, 4.4, 0.4)
        rate_of_turn_data[6] = np.ma.masked
        rate_of_turn_data[16] = np.ma.masked
        rate_of_turn = Parameter('Heading Rate', np.ma.array(rate_of_turn_data))
        airborne = buildsection('Airborne',0,21)
        turning_in_air = TurningInAir()
        turning_in_air.derive(rate_of_turn, airborne)
        expected = buildsections('Turning In Air',[0, 6],[16,21])
        self.assertEqual(turning_in_air.get_slices(), expected.get_slices())


class TestTurningOnGround(unittest.TestCase):
    def test_can_operate(self):
        expected = [('Heading Rate', 'Taxiing')]
        self.assertEqual(TurningOnGround.get_operational_combinations(), expected)

    def test_turning_on_ground_phase_basic(self):
        rate_of_turn_data = np.ma.arange(-12, 12, 1)
        rate_of_turn = Parameter('Heading Rate', np.ma.array(rate_of_turn_data))
        grounded = buildsection('Grounded', 0, 24)
        turning_on_ground = TurningOnGround()
        turning_on_ground.derive(rate_of_turn, grounded)
        expected = buildsections('Turning On Ground',[0, 7], [18,24])
        self.assertEqual(turning_on_ground.get_slices(), expected.get_slices())

    def test_turning_on_ground_phase_with_mask(self):
        rate_of_turn_data = np.ma.arange(-12, 12, 1)
        rate_of_turn_data[10] = np.ma.masked
        rate_of_turn_data[18] = np.ma.masked
        rate_of_turn = Parameter('Heading Rate', np.ma.array(rate_of_turn_data))
        grounded = buildsection('Grounded', 0, 24)
        turning_on_ground = TurningOnGround()
        turning_on_ground.derive(rate_of_turn, grounded)
        # Masked inside is exclusive of the range outer limits, this behaviour
        # is not consistent with TurningInAir test which is inclusive of the
        # start of the range.
        expected = buildsections('Turning On Ground',[0, 7], [18,24])
        self.assertEqual(turning_on_ground.get_slices(), expected.get_slices())

    def test_turning_on_ground_after_takeoff_inhibited(self):
        rate_of_turn_data = np.ma.arange(-12, 12, 1)
        rate_of_turn_data[10] = np.ma.masked
        rate_of_turn_data[18] = np.ma.masked
        rate_of_turn = Parameter('Heading Rate', np.ma.array(rate_of_turn_data))
        grounded = buildsection('Grounded', 0,10)
        turning_on_ground = TurningOnGround()
        turning_on_ground.derive(rate_of_turn, grounded)
        expected = buildsections('Turning On Ground',[0, 7])
        self.assertEqual(turning_on_ground.get_slices(), expected.get_slices())


class TestDescentToFlare(unittest.TestCase):
    def test_can_operate(self):
        self.assertEqual(DescentToFlare.get_operational_combinations(),
                         [('Descent', 'Altitude AAL For Flight Phases')])

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestEngHotelMode(unittest.TestCase):

    def setUp(self):
        self.node_class = EngHotelMode

    def test_can_operate(self):
        family = Attribute('Family', value='ATR-42')
        available = ['Eng (2) Np', 'Eng (1) N1', 'Eng (2) N1', 'Grounded', 'Propeller Brake']
        self.assertTrue(self.node_class.can_operate(available, family=family))
        family.value = 'B737'
        self.assertFalse(self.node_class.can_operate(available, family=family))

    def test_derive_basic(self):
        range_array = np.arange(0, 50, 5)
        eng2_n1 = Parameter('Eng (2) N1',
                            array=np.ma.concatenate((
                                range_array,
                                [50] * 40,
                                range_array[::-1],)))
        eng1_n1 = Parameter('Eng (1) N1', array=np.ma.concatenate(
            ([0] * 10,
             range_array,
             [50] * 20,
             range_array[::-1],
             [0] * 10,)))
        eng2_np = Parameter('Eng (1) N1',
                            array=np.ma.concatenate((
                                [0] * 20,
                                [100] * 20,
                                [0] * 20,)))
        groundeds = buildsections('Grounded', (0, 22), (38, None))
        prop_brake_values_mapping = {1: 'On', 0: '-'}
        prop_brake = M('Propeller Brake', array=np.ma.array([1] * 16 + [0] * 24 + [1] * 20), values_mapping=prop_brake_values_mapping)

        node = self.node_class()
        node.derive(eng2_np, eng1_n1, eng2_n1, groundeds, prop_brake)

        expected = [Section(name='Eng Hotel Mode', slice=slice(10, 16, None), start_edge=10, stop_edge=16),
                    Section(name='Eng Hotel Mode', slice=slice(42, 50, None), start_edge=42, stop_edge=50)]
        self.assertEqual(list(node), expected)
    
    def test_derive(self):
        eng2_np = load(os.path.join(test_data_path, 'eng_hotel_mode_eng2_np.nod'))
        eng1_n1 = load(os.path.join(test_data_path, 'eng_hotel_mode_eng1_n1.nod'))
        eng2_n1 = load(os.path.join(test_data_path, 'eng_hotel_mode_eng2_n1.nod'))
        prop_brake = load(os.path.join(test_data_path, 'eng_hotel_mode_prop_brake.nod'))
        groundeds = buildsections('Grounded', (0, 9203.6875), (17481.6875, 19011.6875))
        
        node = self.node_class()
        node.derive(eng2_np, eng1_n1, eng2_n1, groundeds, prop_brake)
        
        self.assertEqual(node.get_slices(), [slice(7760, 7784), slice(8248, 8632), slice(17696, 17770)])


class TestGearExtending(unittest.TestCase):
    def setUp(self):
        self.node_class = GearExtending
        
    def test_can_operate(self):
        opts = self.node_class.get_operational_combinations()
        self.assertEqual(opts,[
                         ('Gear Down Selected', 'Gear Down', 'Airborne')])


    def test_derive(self):
        gear_down = M('Gear Down', np.ma.array([1,1,1,1,0,0,0,0,0,1,1,1]),
                      values_mapping={0:'Up', 1:'Down'})
        gear_down_sel = M('Gear Down Selected',
                        np.ma.array([1,1,0,0,0,0,0,1,1,1,1,1]),
                        values_mapping={0:'Up', 1:'Down'})
        airs = buildsection('Airborne', 1, 11)
        
        
        node = self.node_class()
        node.derive(gear_down_sel, gear_down, airs)
        expected=buildsection('Gear Extending', 7, 9)
        self.assertEqual(node.get_slices(), expected.get_slices())


class TestGearExtended(unittest.TestCase):
    def test_can_operate(self):
        self.assertEqual(GearExtended.get_operational_combinations(),
                         [('Gear Down',)])

    def test_basic(self):
        gear = M(
            name='Gear Down',
            array=np.ma.array([1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1]),
            values_mapping={0: 'Up', 1: 'Down'})
        gear_ext = GearExtended()
        gear_ext.derive(gear)
        self.assertEqual(gear_ext[0].slice, slice(0, 5))
        self.assertEqual(gear_ext[1].slice, slice(13,16))


class TestGearRetracted(unittest.TestCase):
    def test_can_operate(self):
        self.assertEqual(GearRetracted.get_operational_combinations(),
                         [('Gear Up',)])

    def test_basic(self):
        gear = M(
            name='Gear Up',
            array=np.ma.array([0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0]),
            values_mapping={0: 'Down', 1: 'Up'})
        gear_ext = GearRetracted()
        gear_ext.derive(gear)
        self.assertEqual(gear_ext[0].slice, slice(5, 14))


class TestGoAround5MinRating(unittest.TestCase):
    def test_can_operate(self):
        self.assertEqual(GoAround5MinRating.get_operational_combinations(),
                         [('Go Around And Climbout', 'Touchdown')])

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestTakeoff5MinRating(unittest.TestCase):
    def test_can_operate(self):
        self.assertEqual(Takeoff5MinRating.get_operational_combinations(),
                         [('Takeoff Acceleration Start',)])

    def test_derive_basic(self):
        toffs = KTI('Takeoff Acceleration Start',
                    items=[KeyTimeInstance(index=100)])
        node = Takeoff5MinRating()
        node.derive(toffs)
        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].slice.start, 100)
        self.assertEqual(node[0].slice.stop, 400)


class TestTakeoffRoll(unittest.TestCase):
    def test_can_operate(self):
        self.assertEqual(TakeoffRoll.get_operational_combinations(),
                         [('Takeoff', 'Takeoff Acceleration Start',),('Takeoff', 'Takeoff Acceleration Start', 'Pitch',)])

    def test_derive(self):
        accel_start = KTI('Takeoff Acceleration Start', items=[
                    KeyTimeInstance(967.92513157006306, 'Takeoff Acceleration Start'),
                ])
        takeoffs = S(items=[Section('Takeoff', slice(953, 995), 953, 995)])
        pitch = load(os.path.join(test_data_path,
                                    'TakeoffRoll-pitch.nod'))
        node = TakeoffRoll()
        node.derive(toffs=takeoffs,
                   acc_starts=accel_start,
                   pitch=pitch)
        self.assertEqual(len(node), 1)
        self.assertAlmostEqual(node[0].slice.start, 967.92, places=1)
        self.assertAlmostEqual(node[0].slice.stop, 990.27, places=1)


class TestTakeoffRollOrRejectedTakeoff(unittest.TestCase):
    def test_can_operate(self):
        self.assertEqual(TakeoffRollOrRejectedTakeoff.get_operational_combinations(),
                         [('Takeoff Roll', 'Rejected Takeoff',)])

    def test_derive(self):
        rolls = buildsections('Takeoff Roll', [5,15], [105.2,115.4])
        rejs = buildsections('Rejected Takeoff', [50,65])
        expected = buildsections('Takeoff Roll Or Rejected Takeoff', [5,15], [50,65], [105.2,115.4])
        phase = TakeoffRollOrRejectedTakeoff()
        phase.derive(rolls, rejs)
        self.assertEqual([s.slice for s in sorted(list(expected))],
                         [s.slice for s in list(expected)])


class TestTakeoffRotation(unittest.TestCase):
    def test_can_operate(self):
        self.assertEqual(TakeoffRotation.get_operational_combinations(),
                         [('Liftoff',)])

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestTwoDegPitchTo35Ft(unittest.TestCase):
    def test_can_operate(self):
        self.assertEqual(TwoDegPitchTo35Ft.get_operational_combinations(),
                         [('Takeoff Roll', 'Takeoff',)])

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')
