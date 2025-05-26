#include "Pythia8/Pythia.h"
#include "TFile.h"
#include "TTree.h"

using namespace Pythia8;

int main() {
    // Initialize Pythia
    Pythia pythia;

    // Enable QCD hard processes (jets)
    pythia.readString("HardQCD:all = on");

    // Turn off prompt photon processes to keep background pure
    pythia.readString("PromptPhoton:all = off");

    // Set center of mass energy (e.g. 13 TeV LHC)
    pythia.readString("Beams:eCM = 13000.");

    // Initialize
    pythia.init();

    // Create ROOT file and tree to save photons (from fragmentation etc.)
    TFile *file = new TFile("diphoton_background.root", "RECREATE");
    TTree *tree = new TTree("Events", "Background photon events");

    float pt1=0, eta1=0, phi1=0;
    float pt2=0, eta2=0, phi2=0;
    tree->Branch("pt1", &pt1);
    tree->Branch("eta1", &eta1);
    tree->Branch("phi1", &phi1);
    tree->Branch("pt2", &pt2);
    tree->Branch("eta2", &eta2);
    tree->Branch("phi2", &phi2);

    const int nEvents = 10000;

    for (int iEvent = 0; iEvent < nEvents; ++iEvent) {
        if (!pythia.next()) continue;

        // Select photons in the event
        std::vector<int> photonIndices;
        for (int i = 0; i < pythia.event.size(); ++i) {
            if (pythia.event[i].id() == 22 && pythia.event[i].isFinal()) {
                photonIndices.push_back(i);
            }
        }

        // Skip events with less than 2 photons
        if (photonIndices.size() < 2) continue;

        // Sort photons by pT descending
        std::sort(photonIndices.begin(), photonIndices.end(),
            [&](int a, int b) {
                return pythia.event[a].pT() > pythia.event[b].pT();
            });

        // Fill tree with leading 2 photons info
        pt1  = pythia.event[photonIndices[0]].pT();
        eta1 = pythia.event[photonIndices[0]].eta();
        phi1 = pythia.event[photonIndices[0]].phi();

        pt2  = pythia.event[photonIndices[1]].pT();
        eta2 = pythia.event[photonIndices[1]].eta();
        phi2 = pythia.event[photonIndices[1]].phi();

        tree->Fill();
    }

    pythia.stat();
    tree->Write();
    file->Close();

    return 0;
}
