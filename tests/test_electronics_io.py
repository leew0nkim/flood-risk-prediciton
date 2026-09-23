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

if __name__=='__main__': unittest.main()
