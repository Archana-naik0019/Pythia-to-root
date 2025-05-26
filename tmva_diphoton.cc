#include <TFile.h>
#include <TTree.h>
#include <TChain.h>
#include <TMVA/Factory.h>
#include <TMVA/DataLoader.h>
#include <TMVA/Tools.h>

void tmva_diphoton() {
    // Initialize TMVA
    TMVA::Tools::Instance();

    // Output file for TMVA results
    TFile* outputFile = TFile::Open("TMVA_diphoton.root", "RECREATE");

    // Create TMVA factory and DataLoader
    TMVA::Factory *factory = new TMVA::Factory("TMVAClassification", outputFile,
        "!V:!Silent:Color:DrawProgressBar:Transformations=I;D;P;G,D:AnalysisType=Classification");

    TMVA::DataLoader *dataloader = new TMVA::DataLoader("dataset");

    // Add variables - input features for classification
    dataloader->AddVariable("pt1");
    dataloader->AddVariable("eta1");
    dataloader->AddVariable("phi1");
    dataloader->AddVariable("pt2");
    dataloader->AddVariable("eta2");
    dataloader->AddVariable("phi2");

    // Load signal and background ROOT files and trees
    TFile *signalFile = TFile::Open("diphoton_signal.root");
    TTree *signalTree = (TTree*)signalFile->Get("diphoton");

    TFile *backgroundFile = TFile::Open("diphoton_background.root");
    TTree *backgroundTree = (TTree*)backgroundFile->Get("Events");

    // Add signal and background trees to dataloader with equal weight
    dataloader->AddSignalTree(signalTree, 1.0);
    dataloader->AddBackgroundTree(backgroundTree, 1.0);

    // Prepare training and testing (50% split)
    dataloader->PrepareTrainingAndTestTree(
        "",           // no cuts
        "SplitMode=Random:NormMode=NumEvents:!V"
    );

    // Book a BDT method
    factory->BookMethod(dataloader, TMVA::Types::kBDT, "BDT",
        "!H:!V:NTrees=850:MinNodeSize=2.5%:MaxDepth=3:BoostType=AdaBoost:"
        "AdaBoostBeta=0.5:SeparationType=GiniIndex:nCuts=20");

    // Train, test, and evaluate
    factory->TrainAllMethods();
    factory->TestAllMethods();
    factory->EvaluateAllMethods();

    // Save results
    outputFile->Close();

    // Cleanup
    delete factory;
    delete dataloader;
    signalFile->Close();
    backgroundFile->Close();

    std::cout << "TMVA classification done. See TMVA_diphoton.root for results." << std::endl;
}
