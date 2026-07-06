include("plain-stock-wpf-common.jl")

const SP500_2014_LAMBDAS = [0.0; collect(LogRange(1.0, 1000.0, 30)); Inf]

config = PlainWPFConfig(
    "sp500_2014",
    "stock-returns-sp500-2014.csv",
    env_string("STOCK_WPF_OUTPUT", "plain-stock-wpf-sp500-2014-results.csv"),
    ["l1", "linf"],
    SP500_2014_LAMBDAS,
    "paper30",
    0.5,
    0.3,
    [24, 36, 42],
    [6],
    env_int("STOCK_WPF_TEST_LIMIT", 0),
)

run_plain_wpf_experiment(config)
