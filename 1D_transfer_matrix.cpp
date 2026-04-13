#include <iostream>
#include <fstream>
#include <complex>
#include <vector>
#include <cmath>

using namespace std;

typedef complex<double> cd;

const double pi = 3.141592653589793;

// Structure that holds layer permitivities glass: 2.25, metal: -10 +i, air = 1. 
struct Layer {
    cd epsilon;
    double thickness; // meters
};

// Compute kz
cd compute_kz(cd epsilon, double k0, double kx) {
    return sqrt(k0 * k0 * epsilon - kx * kx);
}

// Compute admittance
cd compute_Y(cd epsilon, cd kz, double k0, bool isTE) {
    if (isTE) {
        return kz / k0;
    } else {
        return epsilon * k0 / kz;
    }
}

// 2x2 matrix multiply
vector<vector<cd>> matmul(vector<vector<cd>> A, vector<vector<cd>> B) {
    vector<vector<cd>> C(2, vector<cd>(2));

    C[0][0] = A[0][0]*B[0][0] + A[0][1]*B[1][0];
    C[0][1] = A[0][0]*B[0][1] + A[0][1]*B[1][1];
    C[1][0] = A[1][0]*B[0][0] + A[1][1]*B[1][0];
    C[1][1] = A[1][0]*B[0][1] + A[1][1]*B[1][1];

    return C;
}

//transfer matrix
vector<vector<cd>> layer_matrix(cd kz, cd Y, double d) {
    cd phi = kz * d;

    vector<vector<cd>> M(2, vector<cd>(2));

    M[0][0] = cos(phi);
    M[0][1] = cd(0,1) * sin(phi) / Y;
    M[1][0] = cd(0,1) * Y * sin(phi);
    M[1][1] = cos(phi);

    return M;
}

int main() {

    // vacuum wavelength, wave number
    double lambda0 = 1e-6;
    double k0 = 2 * pi / lambda0;

    // Define layers
    vector<Layer> layers = {
        {cd(2.25,0), 0},           // glass (incident medium)
        {cd(-10,1), 25e-9},        // metal, nonzero finite thickness
        {cd(1,0), 0}               // air
    };

    ofstream file("tmm_results.csv");
    file << "theta_deg,R_TE,T_TE,R_TM,T_TM\n";

    for (double theta_deg = 0; theta_deg <= 80; theta_deg += 0.5) {

        double theta = theta_deg * pi / 180.0;

        double n0 = sqrt(real(layers[0].epsilon));
        double kx = k0 * n0 * sin(theta);

        for (int pol = 0; pol < 2; pol++) {

            bool isTE = (pol == 0);

            // Initial identity matrix
            vector<vector<cd>> M = {
                {1,0},
                {0,1}
            };

            // Loop through internal layers (exclude first & last)
            for (int i = 1; i < layers.size()-1; i++) {

                cd kz = compute_kz(layers[i].epsilon, k0, kx);
                cd Y  = compute_Y(layers[i].epsilon, kz, k0, isTE);

                auto Mi = layer_matrix(kz, Y, layers[i].thickness);
                M = matmul(M, Mi);
            }

            // Incident & substrate
            cd kz0 = compute_kz(layers[0].epsilon, k0, kx);
            cd Y0  = compute_Y(layers[0].epsilon, kz0, k0, isTE);

            cd kzs = compute_kz(layers.back().epsilon, k0, kx);
            cd Ys  = compute_Y(layers.back().epsilon, kzs, k0, isTE);

            cd numerator = (M[0][0] + M[0][1]*Ys)*Y0 - (M[1][0] + M[1][1]*Ys);
            cd denominator = (M[0][0] + M[0][1]*Ys)*Y0 + (M[1][0] + M[1][1]*Ys);

            cd r = numerator / denominator;
            cd t = (2.0 * Y0) / denominator;

            double R = norm(r);
            double T = real(Ys / Y0) * norm(t);

            if (isTE) {
                file << theta_deg << "," << R << "," << T << ",";
            } else {
                file << R << "," << T << "\n";
            }
        }
    }

    file.close();

    cout << "Done. Output written to tmm_results.csv\n";

    return 0;
}