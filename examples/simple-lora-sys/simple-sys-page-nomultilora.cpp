#include "simple-lora-sys-common.hpp"

int main() {
    return run_sys_experiment(
            "page_nomultilora",
            "page_nomultilora.csv",
            true,
            false);
}