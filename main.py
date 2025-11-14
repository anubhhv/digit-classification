#include <iostream>
#include <fstream>
#include <string>
using namespace std;

struct Subscription {
    int id;
    string name;
    float cost;
    string nextDate;
    bool active;
};

Subscription subs[50];   // simple array
int countSubs = 0;

// ---- Load data from file ----
void loadData() {
    ifstream fin("subs.txt");
    if (!fin.is_open()) return;

    countSubs = 0;
    while (!fin.eof()) {
        fin >> subs[countSubs].id;
        fin >> subs[countSubs].name;
        fin >> subs[countSubs].cost;
        fin >> subs[countSubs].nextDate;
        fin >> subs[countSubs].active;

        if (fin.fail()) break; // avoid duplication on last line
        countSubs++;
    }

    fin.close();
}

// ---- Save data to file ----
void saveData() {
    ofstream fout("subs.txt");

    for (int i = 0; i < countSubs; i++) {
        fout << subs[i].id << " "
             << subs[i].name << " "
             << subs[i].cost << " "
             << subs[i].nextDate << " "
             << subs[i].active << endl;
    }

    fout.close();
}

int main() {

    loadData();  // load existing data

    int choice;

    while (true) {
        cout << "\n===== Subscription Tracker =====\n";
        cout << "1. Add Subscription\n";
        cout << "2. View Subscriptions\n";
        cout << "3. Cancel Subscription\n";
        cout << "4. Exit\n";
        cout << "Enter choice: ";
        cin >> choice;

        if (choice == 1) {
            // Add
            subs[countSubs].id = countSubs + 1;

            cout << "Enter name: ";
            cin >> subs[countSubs].name;

            cout << "Enter cost: ";
            cin >> subs[countSubs].cost;

            cout << "Enter next billing date (YYYY-MM-DD): ";
            cin >> subs[countSubs].nextDate;

            subs[countSubs].active = true;

            countSubs++;
            saveData();  // save changes
            cout << "Subscription added!\n";
        }

        else if (choice == 2) {
            // View
            if (countSubs == 0) {
                cout << "No subscriptions found.\n";
            } else {
                cout << "\n--- All Subscriptions ---\n";
                for (int i = 0; i < countSubs; i++) {
                    cout << "ID: " << subs[i].id << endl;
                    cout << "Name: " << subs[i].name << endl;
                    cout << "Cost: " << subs[i].cost << endl;
                    cout << "Next Billing: " << subs[i].nextDate << endl;
                    cout << "Status: " << (subs[i].active ? "Active" : "Cancelled") << endl;
                    cout << "-------------------------\n";
                }
            }
        }

        else if (choice == 3) {
            // Cancel
            int cid;
            cout << "Enter ID to cancel: ";
            cin >> cid;

            bool found = false;
            for (int i = 0; i < countSubs; i++) {
                if (subs[i].id == cid) {
                    subs[i].active = false;
                    found = true;
                    saveData();
                    cout << "Subscription cancelled!\n";
                    break;
                }
            }

            if (!found) {
                cout << "Invalid ID.\n";
            }
        }

        else if (choice == 4) {
            cout << "Exiting...\n";
            break;
        }

        else {
            cout << "Invalid choice!\n";
        }
    }

    return 0;
}