//
// Created by dn277127 on 2/13/25.
//
#include <iostream>
#include <TCanvas.h>
#include <TF1.h>
#include <TGraph.h>
#include <TMath.h>
#include <TAxis.h>

// Define the 6th order polynomial function
double poly6(double *x, double *par) {
    return TMath::Power(x[0], 6) - 0.14 * TMath::Power(x[0], 5) - 5 * TMath::Power(x[0], 4)
           + 6 * TMath::Power(x[0], 3) + 2 * TMath::Power(x[0], 2) - 3 * x[0] + 1;
}

// Define the 6th order polynomial fitting function
double poly6_fit(double *dummy_x, double *x) {
    return TMath::Power(x[0], 6) - 0.14 * TMath::Power(x[0], 5) - 5 * TMath::Power(x[0], 4)
           + 6 * TMath::Power(x[0], 3) + 2 * TMath::Power(x[0], 2) - 3 * x[0] + 1;
}

void root_local_global_minimizer_check() {
    // Define function with one parameter (dummy parameter for compatibility)
    TF1 *f = new TF1("poly6", poly6, -3, 3, 0);
    TF1 *f_fit = new TF1("poly6_fit", poly6_fit, -3, 3, 1);

    // Create canvas for plotting
    TCanvas *c1 = new TCanvas("c1", "6th Order Polynomial", 800, 600);
    f->SetLineColor(kBlue);
    f->SetTitle("6th Order Polynomial with Global and Local Minimum; x; f(x)");
    f->Draw();

    // Create dummy TGraph data of [0], [-10]
    TGraph *g = new TGraph(1);
    g->SetPoint(0, 0, -10);

    // Plot this TGraph in a second canvas
    TCanvas *c2 = new TCanvas("c2", "Local Minimizer", 800, 600);
    g->SetMarkerStyle(20);
    g->SetMarkerColor(kRed);
    g->Draw("AP");

    // Fit the TGraph with the 6th order polynomial
    float p0 = -2.0;
    f_fit->SetParameter(0, p0);
    g->Fit("poly6_fit", "R");

    // Get the fit parameters
    double *fit_params = f_fit->GetParameters();
    double fit_p0 = fit_params[0];
    std::cout << "Fit parameter p0: " << fit_p0 << std::endl;

    // Update canvas
    c1->Update();
    c2->Update();
    std::cout << "donzo" << std::endl;
}