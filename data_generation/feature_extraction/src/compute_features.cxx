// Compute jet and event-level features from LHCO-style particle data.
//
// Usage: parse_data_parallel_allfeatures_CY.run <input.root> <output.root> <tree_name> [max_events]
//   input tree : data[2100]/D (700 particles x pT, eta, phi, zero-padded), type/I
//   output tree: one entry per input event, same order (can be used as a friend tree)
//   max_events : optional, default all events (for quick tests)
//   threads    : set with OMP_NUM_THREADS
//
// Jets: anti-kT R = 1.0. Event jet: all particles clustered into one jet.
// RSD: RecursiveSoftDrop(beta = 0.7 z_cut = 0.15 infinite depth, R0 = 1.0, dynamical R0).
//
// Output branches: contains many observables (not just the ones used in the paper)
//


#include <iostream>
#include <fstream>
#include <string>
#include <cmath>
#include <cstring>
#include <functional>
#include <algorithm>
#include <exception>
#include <numeric>

#include <fastjet/PseudoJet.hh>
#include <fastjet/JetDefinition.hh>
#include <fastjet/ClusterSequence.hh>

#include "fastjet/contrib/SoftDrop.hh"
#include "fastjet/contrib/RecursiveSoftDrop.hh"
#include "fastjet/contrib/Nsubjettiness.hh"
#include "fastjet/contrib/MeasureDefinition.hh"
#include "fastjet/contrib/AxesDefinition.hh"

#include "TFile.h"
#include "TTree.h"
#include "TChain.h"
#include "TStopwatch.h"
#include "TROOT.h"

#include <omp.h>

//=======================================================
//  main function 
//=======================================================
int main(int argc, char** argv){

    // Expect 3 arguments: input ROOT file, output ROOT file, tree name (+ optional max events)
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0]
                  << " <input_root_file> <output_root_file> <tree_name> [max_events]" << std::endl;
        return 1;
    }

    const char* input_filename = argv[1];
    const char* output_filename = argv[2];
    const char* tree_name = argv[3];
    const Long64_t max_events = (argc > 4) ? std::atoll(argv[4]) : -1;  // -1 = all events

    // threads open the input file in parallel: protect ROOT's global lists
    ROOT::EnableThreadSafety();

    std::cout << "Input ROOT file:  " << input_filename << std::endl;
    std::cout << "Output ROOT file: " << output_filename << std::endl;
    std::cout << "Tree name:        " << tree_name << std::endl;


    // tree for saving derived outputs
    TFile* file = new TFile(output_filename, "recreate");
    TTree* tree_output = new TTree(tree_name, tree_name);

    tree_output->SetAutoFlush(100000);    // flush every 100k events
    tree_output->SetAutoSave(10000000);   // autosave every 10 MB
    
    // variables
    double ht = 0.;
    tree_output->Branch("HT", &ht, "HT/D");
    double mj1 = 0.;
    tree_output->Branch("mj1", &mj1, "mj1/D");
    double mj2 = 0.; 
    tree_output->Branch("mj2", &mj2, "mj2/D");
    double mj1j2 = 0.;
    tree_output->Branch("mj1j2", &mj1j2, "mj1j2/D");
    double mj2j3 = 0.;
    tree_output->Branch("mj2j3", &mj2j3, "mj2j3/D");
    double mj1j2j3 = 0.;
    tree_output->Branch("mj1j2j3", &mj1j2j3, "mj1j2j3/D");

    int njet = 0;
    tree_output->Branch("njet", &njet, "njet/I");
    int njet_100 = 0;
    tree_output->Branch("njet_100", &njet_100, "njet_100/I");
  
    double tau1 = 0.;
    tree_output->Branch("tau1", &tau1, "tau1/D");
    double tau2 = 0.;
    tree_output->Branch("tau2", &tau2, "tau2/D");
    double tau3 = 0.;
    tree_output->Branch("tau3", &tau3, "tau3/D");
    double tau4 = 0.;
    tree_output->Branch("tau4", &tau4, "tau4/D");
    
    double m_all = 0.;
    tree_output->Branch("m_all", &m_all, "m_all/D");
    
    // RSD variables
    double m_rsd = 0.;
    tree_output->Branch("m_rsd", &m_rsd, "m_rsd/D");
    double ht_rsd = 0.;
    tree_output->Branch("ht_rsd", &ht_rsd, "ht_rsd/D");
    double mj1_rsd = 0.;
    tree_output->Branch("mj1_rsd", &mj1_rsd, "mj1_rsd/D");
    double mj2_rsd = 0.;
    tree_output->Branch("mj2_rsd", &mj2_rsd, "mj2_rsd/D");
    double mj1j2_rsd = 0.;
    tree_output->Branch("mj1j2_rsd", &mj1j2_rsd, "mj1j2_rsd/D");
    double mj2j3_rsd = 0.;
    tree_output->Branch("mj2j3_rsd", &mj2j3_rsd, "mj2j3_rsd/D");
    double mj1j2j3_rsd = 0.;
    tree_output->Branch("mj1j2j3_rsd", &mj1j2j3_rsd, "mj1j2j3_rsd/D");
    
    int njet_rsd = 0;
    tree_output->Branch("njet_rsd", &njet_rsd, "njet_rsd/I");
    int njet_100_rsd = 0;
    tree_output->Branch("njet_100_rsd", &njet_100_rsd, "njet_100_rsd/I");
    
    double tau1_rsd = 0.;
    tree_output->Branch("tau1_rsd", &tau1_rsd, "tau1_rsd/D");
    double tau2_rsd = 0.;
    tree_output->Branch("tau2_rsd", &tau2_rsd, "tau2_rsd/D");
    double tau3_rsd = 0.;
    tree_output->Branch("tau3_rsd", &tau3_rsd, "tau3_rsd/D");
    double tau4_rsd = 0.;
    tree_output->Branch("tau4_rsd", &tau4_rsd, "tau4_rsd/D");

    double mjmin = 0.;
    tree_output->Branch("mjmin", &mjmin, "mjmin/D");
    double mjmax = 0.;
    tree_output->Branch("mjmax", &mjmax, "mjmax/D");
    double tau21min = 0.;
    tree_output->Branch("tau21min", &tau21min, "tau21min/D");
    double tau21max = 0.;
    tree_output->Branch("tau21max", &tau21max, "tau21max/D");
    double delta_mjj = 0.;
    tree_output->Branch("delta_mjj", &delta_mjj, "delta_mjj/D");

    int type_new = 0;
    tree_output->Branch("type", &type_new, "type/I");
    
    TStopwatch watch;
    #pragma omp parallel
    {
        //std::cout << "thread id: " << omp_get_thread_num() << std::endl;
        //std::cout << "check 1 " << std::endl;

        // load file
        //TClonesArray* branch_particle = reader.UseBranch((char*)"Particle");
        TChain* chain;
        #pragma omp critical
        {
            //std::cout << "check 2 " << std::endl;
            chain = new TChain(tree_name);
            chain->Add(input_filename);
        }

        std::vector<double> vec_data(2100, 1.);
        int type = 0;

        chain->SetBranchAddress("data", &(vec_data[0])); //connects the branch data to vec_data
        chain->SetBranchAddress("type", &type);          //connects the branch type so ROOT will update type
        
        std::vector<fastjet::PseudoJet> vec_pseudojet(700, fastjet::PseudoJet());
                
        // aux 
        double R_fj1 = 1.0;
        fastjet::JetDefinition jet_def_fj1(fastjet::antikt_algorithm, R_fj1);
        double R_ev = 999.9;
        fastjet::JetDefinition jet_def_ev(fastjet::antikt_algorithm, R_ev);
        
        // rsd groomer
        double z_cut = 0.15;
        double beta_rsd  = 0.7;
        int n=-1; // number of layers (-1 <> infinite)
        double R_rsd = 1.0;
        fastjet::contrib::RecursiveSoftDrop rsd(beta_rsd, z_cut, n, R_rsd);
        rsd.set_dynamical_R0();

        // 4-subjettiness calc
        double beta = 1.0;
        fastjet::contrib::Nsubjettiness calc_jettiness1(
            1, 
            fastjet::contrib::OnePass_WTA_KT_Axes(),     // notice difference with the zenodo distribution
            //fastjet::contrib::OnePass_KT_Axes(),           //after discussion with Marie, used KT axes
            fastjet::contrib::UnnormalizedMeasure(beta)    //before finding out CMS uses normalized measure
            //fastjet::contrib::NormalizedMeasure(beta,1.0)      //CMS uses normalized measure

        );
        fastjet::contrib::Nsubjettiness calc_jettiness2(
            2, 
            fastjet::contrib::OnePass_WTA_KT_Axes(), 
            //fastjet::contrib::OnePass_KT_Axes(),           
            fastjet::contrib::UnnormalizedMeasure(beta)
            //fastjet::contrib::NormalizedMeasure(beta,1.0)
        );
        fastjet::contrib::Nsubjettiness calc_jettiness3(
            3, 
            fastjet::contrib::OnePass_WTA_KT_Axes(), 
            //fastjet::contrib::OnePass_KT_Axes(),           
            fastjet::contrib::UnnormalizedMeasure(beta)
            //fastjet::contrib::NormalizedMeasure(beta,1.0)
        );
        
        fastjet::contrib::Nsubjettiness calc_jettiness4(
            4, 
            fastjet::contrib::OnePass_WTA_KT_Axes(), 
            //fastjet::contrib::OnePass_KT_Axes(),           
            fastjet::contrib::UnnormalizedMeasure(beta)
            //fastjet::contrib::NormalizedMeasure(beta,1.0)
        );

        
        //int nEntries = chain->GetEntries();
        Long64_t totalEntries = chain->GetEntries();
        //Long64_t totalEntries = 1000000; //for testing 
        Long64_t maxEntries = (max_events >= 0) ? std::min(totalEntries, max_events) : totalEntries;

        const Long64_t chunkSize = 100000;

        #pragma omp for ordered schedule(static,1)

        for (Long64_t start = 0; start < maxEntries; start += chunkSize)
        {
            Long64_t end = std::min(start + chunkSize, maxEntries);

            std::cout << "Processing chunk: "
                    << start << " to " << end
                    << " out of " << maxEntries << std::endl;

            for (Long64_t i = start; i < end; ++i)
            {
            //for(int i=0; i<nEntries; ++i){
            //for(int i=0; i<2000; ++i){
            //#pragma omp critical
            //{
                chain->GetEntry(i); 
                //}
                //std::cout << vec_data[0] << " " << vec_data[1] << " " << vec_data[2] << std::endl;
                auto it_data = vec_data.begin();
                auto it_pseudojet_end = vec_pseudojet.begin();

                if (i % 1000 == 0) {
                    std::cout << "Processing entry " << i << " / " << maxEntries << std::endl;
                }

                // reset buffers
                double buf_ht = 0.;
                double buf_mj1 = 0.;
                double buf_mj2 = 0.;
                double buf_mj1j2 = 0.;
                double buf_mj2j3 = 0.;
                double buf_mj1j2j3 = 0.;
                double buf_tau1 = 0.;
                double buf_tau2 = 0.;
                double buf_tau3 = 0.;
                double buf_tau4 = 0.;
                int buf_njet = 0;
                int buf_njet_100 = 0;

                double buf_m_all = 0.;
                double buf_m_rsd = 0.;
                double buf_ht_rsd = 0.;
                double buf_mj1_rsd = 0.;
                double buf_mj2_rsd = 0.;
                double buf_mj1j2_rsd = 0.;
                double buf_mj2j3_rsd = 0.;
                double buf_mj1j2j3_rsd = 0.;
                int buf_njet_rsd = 0;
                int buf_njet_100_rsd = 0;

                double buf_tau1_rsd = 0.;
                double buf_tau2_rsd = 0.;
                double buf_tau3_rsd = 0.;
                double buf_tau4_rsd = 0.;

                double buf_mjmin = 0.;
                double buf_mjmax = 0.;
                double buf_tau21min = 0.;
                double buf_tau21max = 0.;
                double buf_delta_mjj = 0.;

                int buf_type = 0;

                buf_type = type;
                
                // loop over particles
                while(it_data != vec_data.end()){
                    if(it_data[0] == 0.){
                        break;
                    }
                    it_pseudojet_end->reset_momentum_PtYPhiM(
                        it_data[0],
                        it_data[1],
                        it_data[2]
                    );
                    // ht = sum pt
                    buf_ht += it_data[0];
                    
                    it_data=it_data+3;
                    ++it_pseudojet_end;
                }
                
                // printing validation info...
                fastjet::PseudoJet p_total = std::accumulate(vec_pseudojet.begin(), it_pseudojet_end, fastjet::PseudoJet());
                if(i<10){
                    std::cout << p_total.px() << " " << p_total.py() << " " << p_total.pz() << " " << p_total.e() << " " << p_total.m() << " " << std::endl;
                }
                
                // clustering jet WITH R = 1
                fastjet::ClusterSequence cs(
                    std::vector<fastjet::PseudoJet>(
                        vec_pseudojet.begin(),
                        it_pseudojet_end
                    ), 
                    jet_def_fj1
                );
                std::vector<fastjet::PseudoJet> vec_jets = fastjet::sorted_by_pt(cs.inclusive_jets()); //  AK1 jets PT sorted
                buf_njet = cs.inclusive_jets(20.).size();  // Njets soft PT >20
                buf_njet_100 = cs.inclusive_jets(100.).size();  // Njets hard PT >100
                
                // clustering all ev
                fastjet::ClusterSequence cs_ev(
                    std::vector<fastjet::PseudoJet>(
                        vec_pseudojet.begin(),
                        it_pseudojet_end
                    ), 
                    jet_def_ev
                );
                std::vector<fastjet::PseudoJet> vec_event = fastjet::sorted_by_pt(cs_ev.inclusive_jets()); //Event level Jet
                if(vec_event.size() != 1){
                    std::cout << "ERROR: event clustering" << std::endl;
                    throw std::exception();
                }


                // Apply RSD to event jet
                fastjet::PseudoJet p_rsd_ev = rsd(vec_event[0]); //RSD groomed jet from event jet
                if (p_rsd_ev.has_structure()) {
                    for (auto& c : p_rsd_ev.constituents()) {
                        buf_ht_rsd += c.pt();
                    }
                } else {
                    std::cerr << "Warning: p_rsd_ev has no structure, using pt() instead.\n";
                    buf_ht_rsd += p_rsd_ev.pt();
                }
                
                //take a AK1 jet and apply RSD to each jet      
                std::vector<fastjet::PseudoJet> vec_jets_rsd;
                for (auto& jet : vec_jets) {
                    fastjet::PseudoJet jet_rsd = rsd(jet);  
                    vec_jets_rsd.push_back(jet_rsd);        // Save groomed jet
                }

                vec_jets_rsd = fastjet::sorted_by_pt(vec_jets_rsd);
                
                for (auto& jet : vec_jets_rsd) {
                    if (jet.pt() > 20.)  buf_njet_rsd++;
                    if (jet.pt() > 100.) buf_njet_100_rsd++;
                }

                //jet level features
                fastjet::PseudoJet p_j1, p_j2, p_j1j2, p_j2j3, p_j1j2j3;
                if(vec_jets.size() >= 3){
                    p_j1     = vec_jets[0];
                    p_j2     = vec_jets[1];
                    p_j1j2   = vec_jets[0] + vec_jets[1];
                    p_j2j3   = vec_jets[1] + vec_jets[2];
                    p_j1j2j3 = p_j1j2 + vec_jets[2];
                }else if(vec_jets.size() == 2){
                    p_j1     = vec_jets[0];
                    p_j2     = vec_jets[1];
                    p_j1j2   = vec_jets[0] + vec_jets[1];
                    p_j2j3   = vec_jets[1];
                    p_j1j2j3 = p_j1j2;
                }else if(vec_jets.size() == 1){
                    p_j1     = vec_jets[0];
                    p_j1j2   = vec_jets[0];
                    p_j1j2j3 = p_j1j2;
                }else{
                    std::cout << "ERROR: no jets" << std::endl;
                    throw std::exception();
                }
                
                fastjet::PseudoJet p_j1_rsd, p_j2_rsd, p_j1j2_rsd, p_j2j3_rsd, p_j1j2j3_rsd;
                if(vec_jets_rsd.size() >= 3){ 
                    p_j1_rsd     = vec_jets_rsd[0];
                    p_j2_rsd     = vec_jets_rsd[1];
                    p_j1j2_rsd   = vec_jets_rsd[0] + vec_jets_rsd[1];
                    p_j2j3_rsd   = vec_jets_rsd[1] + vec_jets_rsd[2];
                    p_j1j2j3_rsd = p_j1j2_rsd + vec_jets_rsd[2];
                }else if(vec_jets_rsd.size() == 2){
                    p_j1_rsd     = vec_jets_rsd[0];
                    p_j2_rsd     = vec_jets_rsd[1];
                    p_j1j2_rsd   = vec_jets_rsd[0] + vec_jets_rsd[1];
                    p_j2j3_rsd   = vec_jets_rsd[1];
                    p_j1j2j3_rsd = p_j1j2_rsd;
                }else if(vec_jets_rsd.size() == 1){
                    p_j1_rsd     = vec_jets_rsd[0];
                    p_j1j2_rsd   = vec_jets_rsd[0];
                    p_j1j2j3_rsd = p_j1j2_rsd;
                }else{
                    std::cout << "ERROR: no jets _rsd" << std::endl;
                    throw std::exception();
                }
                
                buf_m_all = p_total.m();                              // Invariant mass of total event     

                buf_mj1 = p_j1.m();                                   // Mass of leading jet 
                buf_mj2 = p_j2.m();                                   // Mass of subleading jet
                buf_mj1j2   = p_j1j2.m();                             // Invariant Mass of leading + subleading jet
                buf_mj2j3   = p_j2j3.m();                             // Invariant Mass of subleading + 3rd jet
                buf_mj1j2j3 = p_j1j2j3.m();                           // Invariant Mass of 3 jet system
                buf_tau1 = calc_jettiness1(vec_event[0]);             // tau1 of event jet
                buf_tau2 = calc_jettiness2(vec_event[0]);             // tau2 of event jet  
                buf_tau3 = calc_jettiness3(vec_event[0]);             // tau3 of event jet
                buf_tau4 = calc_jettiness4(vec_event[0]);             // tau4 of event jet

                buf_m_rsd = p_rsd_ev.m();                             // Invariant mass of event level RSD jet

                buf_mj1_rsd = p_j1_rsd.m();                           // Mass of leading RSD jet
                buf_mj2_rsd = p_j2_rsd.m();                           // Mass of subleading RSD jet
                buf_mj1j2_rsd   = p_j1j2_rsd.m();                     // Invariant Mass of leading + subleading RSD jet
                buf_mj2j3_rsd   = p_j2j3_rsd.m();                     // Invariant Mass of subleading + 3rd RSD jet
                buf_mj1j2j3_rsd = p_j1j2j3_rsd.m();                   // Invariant Mass of 3 RSD jet system
                if (p_rsd_ev != 0 && p_rsd_ev.has_constituents()) { 
                    //std::cout << " Test 1" << std::endl;
                    buf_tau1_rsd = calc_jettiness1(p_rsd_ev);             // tau1 of RSD event level jet
                    buf_tau2_rsd = calc_jettiness2(p_rsd_ev);             // tau2 of RSD event level jet
                    buf_tau3_rsd = calc_jettiness3(p_rsd_ev);             // tau3 of RSD event level jet
                    buf_tau4_rsd = calc_jettiness4(p_rsd_ev);             // tau4 of RSD event level jet
                }
                
                //Adding std. cathode features
                // For jet 1
                double tau1_j1 = p_j1.has_structure() ? calc_jettiness1(p_j1) : -1; // tau1 of leading jet
                double tau2_j1 = p_j1.has_structure() ? calc_jettiness2(p_j1) : -1; // tau2 of leading jet
                double tau21_j1 = tau2_j1 / (tau1_j1 + 1e-5);         // tau21 of leading jet

                // For jet 2
                double tau1_j2 = p_j2.has_structure() ? calc_jettiness1(p_j2) : -1; // tau1 of subleading jet
                double tau2_j2 = p_j2.has_structure() ? calc_jettiness2(p_j2) : -1; // tau2 of subleading jet
                double tau21_j2 = tau2_j2 / (tau1_j2 + 1e-5);        // tau21 of subleading jet

                // Sorted according to mj
                buf_tau21min = (buf_mj1 < buf_mj2) ? tau21_j1 : tau21_j2;
                buf_tau21max = (buf_mj1 < buf_mj2) ? tau21_j2 : tau21_j1;
                
                buf_mjmin = std::min(buf_mj1, buf_mj2);  //Mj1
                buf_mjmax = std::max(buf_mj1, buf_mj2);  //Mj2
                buf_delta_mjj = buf_mjmax - buf_mjmin;   //Delta Mjj = Mj2 - Mj1

                // fill tree
                #pragma omp ordered
                {
                    ht = buf_ht;
                    mj1 = buf_mj1;
                    mj2 = buf_mj2;
                    mj1_rsd = buf_mj1_rsd;
                    mj2_rsd = buf_mj2_rsd;
                    mj1j2 = buf_mj1j2;
                    mj2j3 = buf_mj2j3;
                    mj1j2j3 = buf_mj1j2j3;
                    m_all = buf_m_all;
                    m_rsd = buf_m_rsd;
                    ht_rsd = buf_ht_rsd;
                    mj1j2_rsd = buf_mj1j2_rsd;
                    mj2j3_rsd = buf_mj2j3_rsd;
                    mj1j2j3_rsd = buf_mj1j2j3_rsd;
                    njet = buf_njet;
                    njet_100 = buf_njet_100;
                    njet_rsd = buf_njet_rsd;
                    njet_100_rsd = buf_njet_100_rsd;

                    tau1 = buf_tau1;
                    tau2 = buf_tau2; 
                    tau3 = buf_tau3; 
                    tau4 = buf_tau4; 

                    tau1_rsd = buf_tau1_rsd;
                    tau2_rsd = buf_tau2_rsd;
                    tau3_rsd = buf_tau3_rsd;
                    tau4_rsd = buf_tau4_rsd;

                    mjmin = buf_mjmin;
                    mjmax = buf_mjmax;
                    tau21min = buf_tau21min;
                    tau21max = buf_tau21max;                                        
                    delta_mjj = buf_delta_mjj;

                    type_new = buf_type;

                    tree_output->Fill();
                }
            }
            std::cout << "time elapsed: " << watch.RealTime() << "s" << std::endl;
        }
    }
    file->cd();
    tree_output->Write();
    file->Close();

    std::cout << "File saved successfully to "
            << output_filename << std::endl;

    return 0;
}

