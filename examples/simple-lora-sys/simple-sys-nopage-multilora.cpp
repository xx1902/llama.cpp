#include "simple-lora-sys-common.hpp"

int main() {
    return run_sys_experiment(
            "nopage_multilora",
            "nopage_multilora.csv",
            false,
            true);
}