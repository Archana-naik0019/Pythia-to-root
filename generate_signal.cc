#include "Pythia8/Pythia.h"
#include "TFile.h"
#include "TTree.h"
#include <vector>
#include <cmath>

using namespace Pythia8;

int main() {
    // Setup ROOT file and tree
    TFile *outFile = new TFile("diphoton_signal.root", "RECREATE");
    TTree *tree = new TTree("diphoton", "Diphoton Signal Events");

    // Variables to save
    float pt1, eta1, phi1;
    float pt2, eta2, phi2;
    float minv;

    tree->Branch("pt1", &pt1, "pt1/F");
    tree->Branch("eta1", &eta1, "eta1/F");
    tree->Branch("phi1", &phi1, "phi1/F");
    tree->Branch("pt2", &pt2, "pt2/F");
    tree->Branch("eta2", &eta2, "eta2/F");
    tree->Branch("phi2", &phi2, "phi2/F");
    tree->Branch("minv", &minv, "minv/F");

    // Initialize Pythia
    Pythia pythia;
    pythia.readString("Beams:idA = 2212");  // proton
    pythia.readString("Beams:idB = 2212");
    pythia.readString("Beams:eCM = 13000.");

    // Diphoton signal process (direct gamma gamma)
    //pythia.readString("PromptPhoton:qqbar2gammagamma = on");
    pythia.readString("PromptPhoton:all = on");

    pythia.init();

    // Generate events
    int nEvents = 10000;
    for (int i = 0; i < nEvents; ++i) {
        if (!pythia.next()) continue;

        std::vector<Particle> photons;

        for (int j = 0; j < pythia.event.size(); ++j) {
            if (!pythia.event[j].isFinal()) continue;
            if (pythia.event[j].id() == 22) { // photon
                photons.push_back(pythia.event[j]);
            }
        }

        if (photons.size() < 2) continue;

        // Select the two highest-pt photons
        std::sort(photons.begin(), photons.end(), [](const Particle& a, const Particle& b) {
            return a.pT() > b.pT();
        });

        Particle g1 = photons[0];
        Particle g2 = photons[1];

        pt1 = g1.pT();  eta1 = g1.eta();  phi1 = g1.phi();
        pt2 = g2.pT();  eta2 = g2.eta();  phi2 = g2.phi();

        // Invariant mass
        Vec4 p4 = g1.p() + g2.p();
        minv = p4.mCalc();

        tree->Fill();
    }

    pythia.stat();
    tree->Write();
    outFile->Close();

    std::cout << "Saved ROOT file: diphoton_signal.root\n";
    return 0;
}
