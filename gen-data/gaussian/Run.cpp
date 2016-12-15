#include <iostream>
#include <fstream>
#include <string>
using namespace std;

int main(int argc, char** argv )
{
    int line;
    ifstream inf("gaussian_121_60_raw.txt");
    ofstream outf("gaussian_121_60[1-272~265].txt");
    
    if (inf.is_open() && outf.is_open()) {
        while (inf >> line) {
            if (line > 0 && line <= 272) {
                outf << line << '\n';
            } else if (line > 272) {
                outf << 265 - line << '\n';
            }
        }
        inf.close();
        outf.close();
    }

    cout << "Done\n";

    return 0;
}
