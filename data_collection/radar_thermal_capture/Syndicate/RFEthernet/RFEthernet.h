#pragma once

#ifdef _MSC_VER
/*
 * we do not want the warnings about the old deprecated and unsecure CRT functions
 * since these examples can be compiled under *nix as well
 */
#define _CRT_SECURE_NO_WARNINGS
#endif

#include <winsock2.h>

#include "sensor.h"

#include <pcap.h>
#include <tchar.h>
#include <iostream>
#include <thread>
#include <string>
#include <memory>
#include <any>
#include <boost/thread.hpp>

struct RFEthernet : public Sensor 
{
    RFEthernet(ptree::value_type& sensor_settings, ptree::value_type& global_settings);
    ~RFEthernet();

    void AcquireSave(double seconds, boost::barrier& startBarrier);

    pcap_if_t *all_devs;
	pcap_if_t *curr_dev;
	pcap_t *adhandle;
    int _timeout;
    int interface_id;
    std::string cmd_reset_fpga;
    std::string cmd_run_mmwave;
    std::string cmd_close_mmwave;


    void interrupt_pcap_loop(double seconds);

#ifdef _WIN32
    bool LoadNpcapDlls();
#endif
};

/* Callback function invoked by libpcap for every incoming packet */
static void packet_handler(u_char *dumpfile, const struct pcap_pkthdr *header, const u_char *pkt_data);