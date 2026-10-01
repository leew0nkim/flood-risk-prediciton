import json
import unittest
from unittest.mock import Mock
from meit_model.electronics_io import ElectronicsSerial

class ElectronicsIOTests(unittest.TestCase):
    def packet(self):
        return dict(external_depth_m=.164, internal_depth_m=.0,
                    evacuation_complete=False, equalization_active=False)

    def test_json_metres_and_boolean_states(self):
        packet=ElectronicsSerial.parse_sensor_line(json.dumps(self.packet()))
        self.assertEqual(packet.external_depth_m,.164)
        self.assertEqual(packet.internal_depth_m,0.)
        self.assertIs(packet.evacuation_complete,False)
        self.assertIs(packet.equalization_active,False)
        data=self.packet();data['evacuation_complete']=True
        self.assertIs(ElectronicsSerial.parse_sensor_line(json.dumps(data)).evacuation_complete,True)

    def test_invalid_fields_rejected(self):
        for key in ['external_depth_m','internal_depth_m']:
            for invalid in [-1,float('nan'),float('inf'),True,'0.164',None]:
                data=self.packet();data[key]=invalid
                with self.subTest(key=key,value=invalid),self.assertRaises(ValueError):
                    ElectronicsSerial.parse_sensor_line(json.dumps(data))
        for key in ['evacuation_complete','equalization_active']:
            for invalid in [None,0,1,'false']:
                data=self.packet();data[key]=invalid
                with self.subTest(key=key,value=invalid),self.assertRaises(ValueError):
                    ElectronicsSerial.parse_sensor_line(json.dumps(data))
        for line in ['50.66,49.79','[]','null','{}','SAFE']:
            with self.subTest(line=line),self.assertRaises(ValueError):
                ElectronicsSerial.parse_sensor_line(line)

    def test_partial_line_and_invalid_line_recovery(self):
        raw=json.dumps(self.packet()).encode()+b'\n'
        connection=ElectronicsSerial.__new__(ElectronicsSerial)
        connection.serial=Mock()
        connection.serial.readline.side_effect=[raw[:30],b'',raw[30:],b'bad\n',raw]
        self.assertIsNone(connection.read_sensor_packet())
        self.assertIsNone(connection.read_sensor_packet())
        self.assertEqual(connection.read_sensor_packet().external_depth_m,.164)
        self.assertIsNone(connection.read_sensor_packet())
        self.assertEqual(connection.read_sensor_packet().external_depth_m,.164)

    def test_board_messages_are_structured_instead_of_json_errors(self):
        sensor=(json.dumps(self.packet())+'\n').encode()
        connection=ElectronicsSerial.__new__(ElectronicsSerial)
        connection.serial=Mock()
        connection.serial.readline.side_effect=[
            b'Arduino Ready\n', b'SENSOR ERROR\n',
            b'AI DATA: {"state":"SAFE"\n', b'STATE = SAFE\n',
            b'LED = GREEN\n', b'BUZZER = ON\n', b'GATE = OPEN\n', sensor,
        ]
        expected=[
            ('BOARD_READY',None), ('SENSOR_ERROR',None), ('AI_ECHO',None),
            ('BOARD_STATE',None), ('BOARD_LED',None), ('BOARD_BUZZER',None),
            ('BOARD_GATE',None), ('SENSOR_DATA',.164),
        ]
        for event_type,external in expected:
            with self.subTest(event_type=event_type):
                packet=connection.read_sensor_packet()
                self.assertEqual(connection.last_event['type'],event_type)
                self.assertIsNone(connection.last_error)
                self.assertEqual(packet.external_depth_m if packet else None,external)
        status=connection.input_status()
        self.assertTrue(status['board_ready'])
        self.assertEqual(status['board_sensor_status'],'OK')
        self.assertEqual(status['board_reported_state'],'SAFE')
        self.assertEqual(status['board_reported_led'],'GREEN')
        self.assertEqual(status['board_reported_buzzer'],'ON')
        self.assertEqual(status['board_reported_gate'],'OPEN')
        self.assertEqual(status['board_echo_status'],'TRUNCATED')
        self.assertIsNone(status['serial_input_error'])

    def test_unknown_text_and_malformed_json_remain_real_input_errors(self):
        connection=ElectronicsSerial.__new__(ElectronicsSerial)
        connection.serial=Mock()
        connection.serial.readline.side_effect=[b'random debug\n',b'{bad json}\n']
        self.assertIsNone(connection.read_sensor_packet())
        self.assertEqual(connection.last_event['type'],'UNRECOGNIZED_TEXT')
        self.assertIn('Unrecognized board input',connection.last_error)
        self.assertIsNone(connection.read_sensor_packet())
        self.assertEqual(connection.last_event['type'],'INVALID_SENSOR_JSON')
        self.assertIn('Invalid sensor JSON',connection.last_error)

    def test_timeout_retains_most_recent_real_input_event(self):
        connection=ElectronicsSerial.__new__(ElectronicsSerial)
        connection.serial=Mock()
        connection.serial.readline.side_effect=[b'STATE = WARNING\n',b'']
        self.assertIsNone(connection.read_sensor_packet())
        self.assertEqual(connection.last_event,{'type':'BOARD_STATE','value':'WARNING'})
        self.assertIsNone(connection.read_sensor_packet())
        self.assertEqual(connection.input_status()['serial_input_event'],
                         {'type':'BOARD_STATE','value':'WARNING'})

if __name__=='__main__': unittest.main()
