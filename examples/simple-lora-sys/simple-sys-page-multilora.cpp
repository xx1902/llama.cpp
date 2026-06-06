#include "simple-lora-sys-common.hpp"

int main() {
    return run_sys_experiment(
            "page_multilora",
            "page_multilora.csv",
            true,
            true);
}