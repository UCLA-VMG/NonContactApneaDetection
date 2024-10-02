#include "RFEthernet.h"

RFEthernet::RFEthernet(ptree::value_type& sensor_settings, ptree::value_type& global_settings)
    : Sensor(sensor_settings, global_settings), _timeout(sensor_settings.second.get<int>("timeout")),
	interface_id(sensor_settings.second.get<int>("interface_id")),
	cmd_reset_fpga(sensor_settings.second.get<std::string>("reset_fpga_path")),
	cmd_run_mmwave(sensor_settings.second.get<std::string>("run_mmwave_path")),
	cmd_close_mmwave(sensor_settings.second.get<std::string>("close_mmwave_path"))
{
	// Launch cmd file to configure radar parameters.
	WinExec(cmd_run_mmwave.c_str(), SW_HIDE);
	std::this_thread::sleep_for(std::chrono::seconds(static_cast<int>(60))); // wait 60 seconds for lua script to execute

	int total_interfaces=0;
	char errbuf[PCAP_ERRBUF_SIZE];
	
#ifdef _WIN32
	/* Load Npcap and its functions. */
	if (!LoadNpcapDlls())
	{
		fprintf(stderr, "Couldn't load Npcap\n");
		exit(1);
	}
#endif
    
	/* Retrieve the device list on the local machine */
	if (pcap_findalldevs(&all_devs, errbuf) == -1)
	{
		fprintf(stderr,"Error in pcap_findalldevs: %s\n", errbuf);
		exit(1);
	}
    
    /* Print the list */
    for(curr_dev=all_devs; curr_dev; curr_dev=curr_dev->next)
    {
        std::cout << ++total_interfaces << "." << curr_dev->name;
        if (curr_dev->description)
            std::cout << " (" <<  curr_dev->description << ")" << std::endl; 
        else
            std::cout << " (No description available)" << std::endl;
    }

    if(total_interfaces==0)
    {
        std::cout << std::endl << "No interfaces found! Make sure Npcap is installed." << std::endl;
        exit(1);
    }
    
    if(interface_id < 1 || interface_id > total_interfaces)
    {
        std::cout << std::endl << "Interface number out of range." << std::endl;
        /* Free the device list */
        pcap_freealldevs(all_devs);
        exit(1);
    }
		
	/* Jump to the selected adapter */
    for(curr_dev=all_devs, total_interfaces=0 ; total_interfaces < interface_id-1 ; curr_dev=curr_dev->next, total_interfaces++);
}

#ifdef _WIN32
bool RFEthernet::LoadNpcapDlls()
{
	_TCHAR npcap_dir[512];
	UINT len;
	len = GetSystemDirectory(npcap_dir, 480);
	if (!len) {
		fprintf(stderr, "Error in GetSystemDirectory: %x", GetLastError());
		return false;
	}
	_tcscat_s(npcap_dir, 512, _T("\\Npcap"));
	if (SetDllDirectory(npcap_dir) == 0) {
		fprintf(stderr, "Error in SetDllDirectory: %x", GetLastError());
		return false;
	}
	return true;
}
#endif

void RFEthernet::AcquireSave(double seconds, boost::barrier& startBarrier)
{
	pcap_dumper_t *dumpfile;
	char errbuf[PCAP_ERRBUF_SIZE];
    /* Open the adapter */
	if ((adhandle = pcap_open_live(curr_dev->name,	// name of the device
							 65536,			// portion of the packet to capture. 
											// 65536 grants that the whole packet will be captured on all the MACs.
							 1,				// promiscuous mode (nonzero means promiscuous)
							 _timeout,			// read timeout
							 errbuf			// error buffer
							 )) == NULL)
	{
		fprintf(stderr,"\nUnable to open the adapter. %s is not supported by Npcap\n", curr_dev->name);
		/* Free the device list */
		pcap_freealldevs(all_devs);
		exit(1);
	}

	/* Open the dump file */
    std::ostringstream filename;
    filename << rootPath << sensorName << ".raw";
	dumpfile = pcap_dump_open(adhandle, filename.str().c_str());

	if(dumpfile==NULL)
	{
		fprintf(stderr,"\nError opening output file\n");
		exit(1);
	}
	
    /* At this point, we no longer need the device list. Free it */
    pcap_freealldevs(all_devs);
    
    /* start the capture */
	startBarrier.wait();
	boost::thread interrupt_thread(boost::bind(&RFEthernet::interrupt_pcap_loop, this, seconds));
    pcap_loop(adhandle, 0, packet_handler, (unsigned char *)dumpfile);
	std::cout << "RF Execution Complete" << std::endl;

	interrupt_thread.join();
    pcap_close(adhandle);

	std::cout << "Waiting For RF Stop Frame to be called!" << std::endl;
	std::this_thread::sleep_for(std::chrono::seconds(static_cast<int>(30))); // wait 5400 seconds for main script to execute (give patient chance to sleep)
	std::cout << "Waiting For RF Stop Frame Done!" << std::endl;

	// Launch cmd file to configure radar parameters.
	WinExec(cmd_close_mmwave.c_str(), SW_HIDE);
	std::this_thread::sleep_for(std::chrono::seconds(static_cast<int>(5))); // wait 5 seconds for mmwave studio to close
}

static void packet_handler(u_char *dumpfile, const struct pcap_pkthdr *header, const u_char *pkt_data)
{
	/* save the packet on the dump file */
	pcap_dump(dumpfile, header, pkt_data);
}

void RFEthernet::interrupt_pcap_loop(double seconds)
{
    std::this_thread::sleep_for(std::chrono::seconds(static_cast<int>(seconds)));
	pcap_breakloop(adhandle);
}

RFEthernet::~RFEthernet()
{
}
