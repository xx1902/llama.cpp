#include "simple-lora-sys-common.hpp"

int main() {
    return run_sys_experiment(
            "nopage_nomultilora",
            "nopage_nomultilora.csv",
            false,
            false);
}