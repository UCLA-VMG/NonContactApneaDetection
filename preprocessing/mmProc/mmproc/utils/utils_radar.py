# Copyright 2009 Joshua Wright
# 
# This file is part of gr-bluetooth
# 
# gr-bluetooth is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2, or (at your option)
# any later version.
# 
# gr-bluetooth is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with gr-bluetooth; see the file COPYING.  If not, write to
# the Free Software Foundation, Inc., 51 Franklin Street,
# Boston, MA 02110-1301, USA.

import struct
import time
import numpy as np
import tqdm
from tqdm import tqdm

class PcapReader():
    def __init__(self, savefile):
        '''
        Opens the specified file, validates a libpcap header is present.
        @type savefile: String
        @param savefile: Input libpcap filename to open
        @rtype: None
        '''

        self.pcaph_magic_num = 0xa1b2c3d4
        self.pcaph_ver_major = 2
        self.pcaph_ver_minor = 4
        self.pcaph_thiszone = 0
        self.pcaph_sigfigs = 0
        self.pcaph_snaplen = 65535

        self.pcaph_len = 24
        self.__fh = open(savefile, mode='rb')
        self._pcaphsnaplen = 0
        header = self.__fh.read(self.pcaph_len)

        # Read the first 4 bytes for the magic number, determine endianness
        magicnum = struct.unpack("I", header[0:4])[0]
        if magicnum != 0xd4c3b2a1:
            # Little endian
            self.__endflag = "<"
        elif magicnum == 0xa1b2c3d4:
            # Big endign
            self.__endflag = ">"
        else:
            raise Exception('Specified file is not a libpcap capture')

        pcaph = struct.unpack("%sIHHIIII"%self.__endflag, header)
        if pcaph[1] != self.pcaph_ver_major and pcaph[2] != self.pcaph_ver_minor \
                and pcaph[3] != self.pcaph_thiszone and pcaph[4] != self.pcaph_sigfigs \
                and pcaph[5] != self.pcaph_snaplen:
            raise Exception('Unsupported pcap header format or version')

        self._pcaphsnaplen = pcaph[5]
        self._datalink = pcaph[6]

    def datalink(self):
        return self._datalink

    def close(self):
        '''
        Closes the output packet capture; wrapper for pcap_close().
        @rtype: None
        '''
        self.pcap_close()

    def pcap_close(self):
        '''
        Closes the output packet capture.
        @rtype: None
        '''
        self.__fh.close()

    def pnext(self):
        '''
        Wrapper for pcap_next to mimic method for Daintree SNA
        '''
        return self.pcap_next()
 
    def pcap_next(self):
        '''
        Retrieves the next packet from the capture file.  Returns a list of
        [Hdr, packet] where Hdr is a list of [timestamp, snaplen, plen] and
        packet is a string of the payload content.  Returns None at the end
        of the packet capture.
        @rtype: List
        '''
        # Read the next header block
        PCAPH_RECLEN = 16
        rechdrdata = self.__fh.read(PCAPH_RECLEN)

        try:
            rechdrtmp = struct.unpack("%sIIII"%self.__endflag, rechdrdata)
        except struct.error:
            return [None,None]

        rechdr = [
                float("%s.%s"%(rechdrtmp[0],"0"*(6-len(str(rechdrtmp[1]))) + str(rechdrtmp[1]))), 
                rechdrtmp[2], 
                rechdrtmp[3]
                ]
        if rechdr[1] > rechdr[2] or rechdr[1] > self._pcaphsnaplen or rechdr[2] > self._pcaphsnaplen:
            raise Exception('Corrupted or invalid libpcap record header (included length exceeds actual length)')

        # Read the included packet length
        frame = self.__fh.read(rechdr[1])
        # Drop the first 42 samples from the list
        return [rechdr, frame]


class PcapDumper():
    def __init__(self, datalink, savefile):
        '''
        Creates a libpcap file using the specified datalink type.
        @type datalink: Integer
        @param datalink: Datalink type, one of DLT_* defined in pcap-bpf.h
        @type savefile: String
        @param savefile: Output libpcap filename to open
        @rtype: None
        '''

        self.pcaph_magic_num = 0xa1b2c3d4
        self.pcaph_ver_major = 2
        self.pcaph_ver_minor = 4
        self.pcaph_thiszone = 0
        self.pcaph_sigfigs = 0
        self.pcaph_snaplen = 65535

        self.__fh = open(savefile, mode='wb')
        self.__fh.write(''.join([
            struct.pack("I", self.pcaph_magic_num), 
            struct.pack("H", self.pcaph_ver_major),
            struct.pack("H", self.pcaph_ver_minor),
            struct.pack("I", self.pcaph_thiszone),
            struct.pack("I", self.pcaph_sigfigs),
            struct.pack("I", self.pcaph_snaplen),
            struct.pack("I", datalink)
            ]))

    def pcap_dump(self, packet, ts_sec=None, ts_usec=None, orig_len=None):
        '''
        Appends a new packet to the libpcap file.  Optionally specify ts_sec
        and tv_usec for timestamp information, otherwise the current time is
        used.  Specify orig_len if your snaplen is smaller than the entire
        packet contents.
        @type ts_sec: Integer
        @param ts_sec: Timestamp, number of seconds since Unix epoch.  Default
        is the current timestamp.
        @type ts_usec: Integer
        @param ts_usec: Timestamp microseconds.  Defaults to current timestamp.
        @type orig_len: Integer
        @param orig_len: Length of the original packet, used if the packet you
        are writing is smaller than the original packet.  Defaults to the
        specified packet's length.
        @type packet: String
        @param packet: Packet contents
        @rtype: None
        '''

        if ts_sec == None or ts_usec == None:
            # There must be a better way here that I don't know -JW
            s_sec, s_usec = str(time.time()).split(".")
            ts_sec = int(s_sec)
            ts_usec = int(s_usec)

        if orig_len == None:
            orig_len = len(packet)

        plen = len(packet)

        self.__fh.write(''.join([
            struct.pack("I", ts_sec),
            struct.pack("I", ts_usec),
            struct.pack("I", orig_len),
            struct.pack("I", plen),
            packet
            ]))

        return

    def close(self):
        '''
        Closes the output packet capture; wrapper for pcap_close().
        @rtype: None
        '''
        self.pcap_close()

    def pcap_close(self):
        '''
        Closed the output packet capture.
        @rtype: None
        '''
        self.__fh.close()


class Organizer():
    def __init__(self, file_path: str, radar_config: tuple, verbose=False):
        np.set_printoptions(threshold=np.inf,linewidth=325)
        self.max_packet_size = 4096
        self.bytes_in_packet = 1456
        self.bytes_in_header = 10

        self.all_data = None
        self.timestamps = []
        self.extract_packets(file_path)
        self.data = self.all_data[0]
        self.packet_num = self.all_data[1]
        self.byte_count = self.all_data[2]

        self.num_packets = len(self.byte_count)
        self.num_chirps = radar_config[0]*radar_config[1]
        self.num_rx = radar_config[2]
        self.num_samples = radar_config[3]
        self.radar_config = radar_config

        self.bytes_in_frame = self.num_chirps * self.num_rx * self.num_samples * 2 * 2
        self.bytes_in_frame_clipped = (self.bytes_in_frame // self.bytes_in_packet) * self.bytes_in_packet
        self.uint16_in_frame = self.bytes_in_frame // 2
        self.num_packets_per_frame = self.bytes_in_frame // self.bytes_in_packet

        self.start_time = self.all_data[3]
        self.end_time = self.all_data[4]
        self.verbose = verbose

    def extract_packets(self, file_path):
        data_packets = []
        packet_numbers = []
        byte_counts = []
        packet_arrivals = []

        reader = PcapReader(file_path)
        next_packet = reader.pnext()
        count = 1
        while next_packet[0] is not None:
            curr_time = next_packet[0][0]
            self.timestamps.append(curr_time)
            data_=next_packet[1]
            data = data_[42:]
            if(len(data) == self.bytes_in_header + self.bytes_in_packet): #Assert this is a Radar packet
                packet_num = struct.unpack('<1l', data[:4])[0]
                packet_numbers.append(packet_num)

                byte_count = struct.unpack('>Q', b'\x00\x00' + data[4:10][::-1])[0]
                byte_counts.append(byte_count)

                data = data[10:]
                data_packets.append(np.frombuffer(data, dtype=np.uint16))
            # else:
                # print(len(data))
            next_packet = reader.pnext()
            count+=1
        print("Count:", count)
        print("True Packet Count:", len(data_packets))
        self.all_data = (data_packets, packet_numbers, byte_counts, 0, 0)

    def iq(self, raw_frame):
        """Reorganizes raw ADC data into a full frame
        Args:
            raw_frame (ndarray): Data to format
            num_chirps: Number of chirps included in the frame
            num_rx: Number of receivers used in the frame
            num_samples: Number of ADC samples included in each chirp
        Returns:
            ndarray: Reformatted frame of raw data of shape (num_chirps, num_rx, num_samples)
        """
        ret = np.zeros(len(raw_frame) // 2, dtype=np.csingle)
        ret = ret.reshape((self.num_chirps, self.num_rx, self.num_samples))

        # Separate IQ data
        # ret[0::2] = raw_frame[0::4] + 1j * raw_frame[2::4]
        # ret[1::2] = raw_frame[1::4] + 1j * raw_frame[3::4]
        ret[:,0,:] = np.reshape(raw_frame[0::8] + 1j * raw_frame[4::8], (self.num_chirps, self.num_samples))
        ret[:,1,:] = np.reshape(raw_frame[1::8] + 1j * raw_frame[5::8], (self.num_chirps, self.num_samples))
        ret[:,2,:] = np.reshape(raw_frame[2::8] + 1j * raw_frame[6::8], (self.num_chirps, self.num_samples))
        ret[:,3,:] = np.reshape(raw_frame[3::8] + 1j * raw_frame[7::8], (self.num_chirps, self.num_samples))

        return ret

    def get_frames(self, start_chunk, end_chunk, bc):
        # if first packet received is not the first byte transmitted
        if bc[start_chunk] == 0:
            bytes_left_in_curr_frame = 0
            start = start_chunk*(self.bytes_in_packet // 2)
        else:
            frames_so_far = bc[start_chunk] // self.bytes_in_frame
            bytes_so_far = frames_so_far * self.bytes_in_frame
            # bytes_left_in_curr_frame = bc[start_chunk] - bytes_so_far
            bytes_left_in_curr_frame = (frames_so_far+1)*self.bytes_in_frame - bc[start_chunk]
            start = (bytes_left_in_curr_frame // 2) + start_chunk*(self.bytes_in_packet // 2)
        # find num of frames
        total_bytes = bc[end_chunk] - (bc[start_chunk] + bytes_left_in_curr_frame)
        num_frames = total_bytes // (self.bytes_in_frame)

        frames = np.zeros((num_frames, self.uint16_in_frame), dtype=np.int16)
        ret_frames = np.zeros((num_frames, self.num_chirps, self.num_rx, self.num_samples), dtype=complex)		
        # compress all received data into one byte stream
        all_uint16 = np.array(self.data).reshape(-1)
        # only choose uint16 starting from a new frame
        all_uint16 = all_uint16[start:]
        # organizing into frames
        for i in range(num_frames):
            frame_start_idx = i*self.uint16_in_frame
            frame_end_idx = (i+1)*self.uint16_in_frame
            frame = all_uint16[frame_start_idx:frame_end_idx]
            frames[i][:len(frame)] = frame.astype(np.int16)
            ret_frames[i] = self.iq(frames[i])	

        return ret_frames


    def organize(self):

        if self.verbose: print('Start time: ', self.start_time)
        if self.verbose: print('End time: ', self.end_time)

        self.byte_count = np.array(self.byte_count)
        self.data = np.array(self.data)
        self.packet_num = np.array(self.packet_num)

        bc = np.array(self.byte_count)
        print(self.byte_count.shape)

        packets_ooo = np.where(np.array(self.packet_num[1:])-np.array(self.packet_num[0:-1]) != 1)[0]
        is_not_monotonic = np.where((np.array(self.packet_num[1:])-np.array(self.packet_num[0:-1])) < 0)[0]

        if self.verbose: print('Non monotonic packets: ', is_not_monotonic)

        if len(packets_ooo) == 0:
            if self.verbose: print('packets in order')
            start_chunk = 0
            ret_frames = self.get_frames(start_chunk, -1, bc)

        elif len(packets_ooo) == 1:
            if self.verbose: print('1 packet not in order')
            start_chunk = packets_ooo[0]+1
            ret_frames = self.get_frames(start_chunk, -1, bc)

        else:
            if self.verbose: print('Packet num not in order')
            packets_ooo = np.append(packets_ooo, len(self.packet_num)-1)

            diff = []
            for i in range(len(packets_ooo)-1):
                diff.append(self.packet_num[packets_ooo[i]+1]-self.packet_num[packets_ooo[i]])
            
            packets_lost = np.sum(np.array(diff))

            packets_expected = self.packet_num[-1]-self.packet_num[0]+1
            if self.verbose: print('Total packets lost ', packets_lost)
            if self.verbose: print('Total packets expected ', packets_expected)
            if self.verbose: print('Fraction lost ', packets_lost/packets_expected)

            new_packets_ooo = []
            start_new_packets_ooo = []
            end_new_packets_ooo = []
            for i in range(1, len(packets_ooo)):
                if (packets_ooo[i] - packets_ooo[i-1]) > self.num_packets_per_frame*2:
                    new_packets_ooo.append(packets_ooo[i-1])
                    start_new_packets_ooo.append(packets_ooo[i-1])
                    end_new_packets_ooo.append(packets_ooo[i])

            new_packets_ooo = np.append(new_packets_ooo, -1)

            for i in tqdm(range(len(start_new_packets_ooo))):
                start_chunk = start_new_packets_ooo[i]+1
                end_chunk = end_new_packets_ooo[i]
                curr_frames = self.get_frames(start_chunk, end_chunk, bc)
                if i == 0:
                    ret_frames = curr_frames
                else:
                    ret_frames = np.concatenate((ret_frames, curr_frames), axis=0)

        return ret_frames