# D-Suite Alpha phase WP1.1: Network Models Outputs
Version 1.1, M. Deakin, Jan 2025
Contact: [matthew.deakin@newcastle.ac.uk](emailto:matthew.deakin@newcastle.ac.uk)

For full details about the D-Suite Alpha phase project see the [project link](https://smarter.energynetworks.org/projects/10060423-a-1/).

The network files are `.dss` files. These can be viewed with a text editor or loaded into software such as OpenDSS.

The geographical scale of the network has been deliberately changed for obfuscation purposes. Specifically, all non-trivial linecodes have been increased by a common factor, and the corresponding line lengths been reduced by the same factor. Similarly, the bus xy-coordinates' magnitude has also been scaled by this same factor. The modelled impedance therefore remains unchanged (up to numerical precision) and the shape of the plotted network is retained, but, the unit of the distances is explicitly *not* given.

## Network model context
The network code is `SPx-y`:
- `x` is for service area: 'M' for SP Manweb; 'D', for SP Distribution (see, e.g., [NESO's map](https://www.neso.energy/data-portal/gis-boundaries-gb-dno-license-areas/gb_dno_licence_areas_20200506) for the boundaries of GB serice areas)
- `y` is either 'r', for rural, or 's', 'u' for urban/suburban (in practise no significant difference should be read into 's' vs 'u').

The SPM-u, -s networks are interconnected networks. These type of LV networks are more complicated than networks such as the (radial) IEEE EU LV test network, which is not interconnected test cases. Within the UK, interconnected LV networks are mostly limited to the Manweb region and London. For the substations interconnected with the substation of interest (this can be identified via the energymeter). Not all feeders on the interconnected substations are included as part of this model; only those connecting with that central network.

Other networks may have some loops. These may be artefacts of the vectorization process. In general, monitoring and smart meter data could be used to validate these network models, however, this was beyond the scope of the present work.

Link box open/close status can only be changed manually (i.e., the status cannot be changed remotely). The status of these link boxes is not monitored directly by the network operator.

Validation of these network models has been carried out by considering if voltage drops and cable loading is not excessive under present day conditions.

Without LV (or HV) monitoring, the voltage at the high voltage (11 kV) side of the substation transformer should not be considered a known variable when modelling voltages under increased LCT uptake, and should instead be considered a variable and appropriate sensitivity analysis conducted.

In contrast to previous networks of this type, transformer capacities and reactance have been confirmed by site inspection. Per unit resistances are assumed. The turns ratio implies a +3.75% voltage boost (the UK nominal voltage is 400 V, where the turns ratio is 11/0.415).

The phases for loads are not systematically known. For modelling, these have been assumed to be have been installed with phases rotating (a,b,c,a,b,c,...)

The number of customers per load can be 1 or more than 1 (e.g., for a tower block, there can be many customers). The number of customers can be determined by considering the WP1.1 dataset "D-Suite Substation Modelling: LCT Profile Allocations", in the file `ncl_bus_connection_censored.csv`.

xy co-ordinates are relative to an arbitrary point on the network. As mentioned previously, the units of these positions are explicitly not given.

The SPD-U network has multiple transformers in the model, but is operated
radially. Plotting in OpenDSS will show the parts of the network not part of 
the substation of interest (in pink as these are not downstream of the 
energy meter).

## D-Suite devices

These models have been developed to explore the potential use of D-Suite devices. 

D-Suite devices are power electronic devices (PEDs) that enable power flows and voltages to be controlled to mitigate network congestion caused by low carbon technologies (LCTs) such as electric vehicles, heat pumps, or solar PV. Three devices are considered, Soft Open Points (SOPs), Smart Transformers (STs), and Static Compensators (STATCOMs), each which has a different topology. The project outputs include several [publicly available reports](https://smarter.energynetworks.org/projects/10086622/) on the design, control and benefits of these devices.

Notes on the placement of the D-Suite devices in the network:
- Each device is placed in the network as three (or six) single-phase constant power loads. These can therefore be used for model unbalance power injections from these devices.
- To support identifying these loads, e.g., when working through the COM interface in another language, these have been named as 'sop', 'statcom','str'; they have also had a reactive power associated with them (see master_dsuite.dss files). The out-of-the-box power flow is *not* representative of their use.
- SOPs are placed at normally open points (i.e. open link boxes). If there is only one half of the link box, a single side of the SOP is placed. (Assumptions must then be made as to the network capacity on the other side of the SOP.)
- STATCOMS have been placed by-eye to cover a wide range of network locations.
- STs are placed at the substation of interest. Only the shunt part is given in the model (full shunt-series operation can be implemented in OpenDSS using a UPFC model).