# forecast_benchmark_2026-09-28

Account zones only (India 9, US 36, Europe 21); xLSTM = mean of 5 seeds. Generated from metrics/per_model_per_region.csv.

## carbon, holdout

| model | resource | window | region | n_series | RMSE | MASE | WMAPE_pct | PICP | CRPS |
|---|---|---|---|---|---|---|---|---|---|
| sarima_dcf | carbon | holdout | All | 66.000 | 37.192 | 0.799 | 7.868 | 0.867 | 19.585 |
| chronos_2 | carbon | holdout | All | 66.000 | 38.563 | 0.790 | 8.095 | 0.846 | 19.265 |
| timesfm_2p5 | carbon | holdout | All | 66.000 | 40.573 | 0.835 | 8.410 | 0.884 | 20.555 |
| seasonal_naive | carbon | holdout | All | 66.000 | 41.656 | 0.825 | 8.426 | 0.782 | 20.891 |
| sarima_core | carbon | holdout | All | 66.000 | 42.436 | 0.872 | 8.764 | 0.842 | 21.611 |
| xlstm | carbon | holdout | All | 66.000 | 46.345 | 0.985 | 9.798 |  |  |
| nexus_llm | carbon | holdout | All | 66.000 | 46.840 | 0.957 | 9.660 |  |  |
| sarima_dcf | carbon | holdout | Europe | 21.000 | 41.958 | 0.812 | 14.843 | 0.861 | 23.863 |
| chronos_2 | carbon | holdout | Europe | 21.000 | 42.194 | 0.663 | 14.545 | 0.845 | 22.166 |
| seasonal_naive | carbon | holdout | Europe | 21.000 | 42.416 | 0.682 | 14.542 | 0.794 | 22.658 |
| sarima_core | carbon | holdout | Europe | 21.000 | 42.755 | 0.762 | 15.273 | 0.853 | 23.783 |
| timesfm_2p5 | carbon | holdout | Europe | 21.000 | 43.619 | 0.710 | 15.131 | 0.901 | 23.294 |
| xlstm | carbon | holdout | Europe | 21.000 | 51.219 | 0.834 | 17.574 |  |  |
| nexus_llm | carbon | holdout | Europe | 21.000 | 56.668 | 0.920 | 19.880 |  |  |
| sarima_dcf | carbon | holdout | India | 9.000 | 49.657 | 0.981 | 5.435 | 0.824 | 24.609 |
| chronos_2 | carbon | holdout | India | 9.000 | 57.700 | 1.121 | 6.470 | 0.731 | 26.302 |
| seasonal_naive | carbon | holdout | India | 9.000 | 60.090 | 1.095 | 6.391 | 0.731 | 28.044 |
| timesfm_2p5 | carbon | holdout | India | 9.000 | 60.994 | 1.187 | 6.652 | 0.731 | 28.224 |
| nexus_llm | carbon | holdout | India | 9.000 | 62.142 | 1.267 | 7.000 |  |  |
| xlstm | carbon | holdout | India | 9.000 | 62.753 | 1.368 | 7.249 |  |  |
| sarima_core | carbon | holdout | India | 9.000 | 64.872 | 1.323 | 7.322 | 0.731 | 30.620 |
| chronos_2 | carbon | holdout | US | 36.000 | 29.249 | 0.781 | 6.444 | 0.875 | 15.814 |
| sarima_dcf | carbon | holdout | US | 36.000 | 29.874 | 0.745 | 6.361 | 0.882 | 15.833 |
| timesfm_2p5 | carbon | holdout | US | 36.000 | 31.275 | 0.820 | 6.717 | 0.912 | 17.040 |
| sarima_core | carbon | holdout | US | 36.000 | 34.396 | 0.824 | 7.017 | 0.863 | 18.092 |
| nexus_llm | carbon | holdout | US | 36.000 | 34.404 | 0.901 | 7.080 |  |  |
| seasonal_naive | carbon | holdout | US | 36.000 | 35.059 | 0.842 | 7.064 | 0.787 | 18.071 |
| xlstm | carbon | holdout | US | 36.000 | 37.712 | 0.978 | 8.050 |  |  |

## carbon, rolling

| model | resource | window | region | n_series | RMSE | MASE | WMAPE_pct | PICP | CRPS |
|---|---|---|---|---|---|---|---|---|---|
| sarima_dcf | carbon | rolling | All | 66.000 | 41.824 | 0.923 | 8.974 | 0.816 | 21.893 |
| chronos_2 | carbon | rolling | All | 66.000 | 42.695 | 0.914 | 9.136 | 0.815 | 21.682 |
| sarima_core | carbon | rolling | All | 66.000 | 44.435 | 0.945 | 9.388 | 0.815 | 22.911 |
| timesfm_2p5 | carbon | rolling | All | 66.000 | 45.273 | 0.975 | 9.663 | 0.834 | 23.252 |
| seasonal_naive | carbon | rolling | All | 66.000 | 46.877 | 1.007 | 10.060 | 0.725 | 24.417 |
| xlstm | carbon | rolling | All | 66.000 | 51.395 | 1.126 | 11.275 |  |  |
| sarima_dcf | carbon | rolling | Europe | 21.000 | 52.038 | 0.955 | 17.264 | 0.782 | 28.906 |
| chronos_2 | carbon | rolling | Europe | 21.000 | 53.555 | 0.890 | 17.482 | 0.782 | 27.794 |
| timesfm_2p5 | carbon | rolling | Europe | 21.000 | 55.836 | 0.945 | 18.377 | 0.815 | 29.378 |
| sarima_core | carbon | rolling | Europe | 21.000 | 56.759 | 0.954 | 18.368 | 0.784 | 29.778 |
| seasonal_naive | carbon | rolling | Europe | 21.000 | 58.123 | 0.984 | 19.366 | 0.690 | 31.422 |
| xlstm | carbon | rolling | Europe | 21.000 | 62.634 | 1.059 | 20.826 |  |  |
| sarima_dcf | carbon | rolling | India | 9.000 | 43.555 | 1.177 | 4.765 | 0.829 | 21.009 |
| sarima_core | carbon | rolling | India | 9.000 | 43.841 | 1.169 | 4.844 | 0.801 | 21.981 |
| timesfm_2p5 | carbon | rolling | India | 9.000 | 44.268 | 1.045 | 4.393 | 0.810 | 20.453 |
| chronos_2 | carbon | rolling | India | 9.000 | 45.166 | 1.013 | 4.605 | 0.819 | 20.190 |
| seasonal_naive | carbon | rolling | India | 9.000 | 46.242 | 1.151 | 4.810 | 0.745 | 21.959 |
| xlstm | carbon | rolling | India | 9.000 | 50.588 | 1.355 | 6.046 |  |  |
| sarima_dcf | carbon | rolling | US | 36.000 | 34.137 | 0.862 | 7.110 | 0.834 | 17.949 |
| chronos_2 | carbon | rolling | US | 36.000 | 34.306 | 0.912 | 7.344 | 0.833 | 18.366 |
| sarima_core | carbon | rolling | US | 36.000 | 35.437 | 0.902 | 7.366 | 0.836 | 19.059 |
| timesfm_2p5 | carbon | rolling | US | 36.000 | 37.969 | 0.981 | 7.948 | 0.850 | 20.145 |
| seasonal_naive | carbon | rolling | US | 36.000 | 38.968 | 0.997 | 8.120 | 0.742 | 20.741 |
| xlstm | carbon | rolling | US | 36.000 | 43.671 | 1.127 | 9.237 |  |  |

## electricity, holdout

| model | resource | window | region | n_series | RMSE | MASE | WMAPE_pct | PICP | CRPS |
|---|---|---|---|---|---|---|---|---|---|
| seasonal_naive | electricity | holdout | All | 66.000 | 821.882 | 0.939 | 5.763 | 0.721 | 396.292 |
| chronos_2 | electricity | holdout | All | 66.000 | 879.001 | 0.836 | 5.664 | 0.780 | 385.629 |
| timesfm_2p5 | electricity | holdout | All | 66.000 | 898.869 | 0.879 | 5.830 | 0.817 | 398.895 |
| sarima_core | electricity | holdout | All | 66.000 | 907.262 | 0.958 | 5.982 | 0.804 | 407.952 |
| sarima_dcf | electricity | holdout | All | 47.000 | 976.802 | 0.922 | 6.907 | 0.812 | 427.667 |
| nexus_llm | electricity | holdout | All | 66.000 | 1015.508 | 1.088 | 6.741 |  |  |
| xlstm | electricity | holdout | All | 66.000 | 1089.322 | 1.067 | 7.000 |  |  |
| seasonal_naive | electricity | holdout | Europe | 21.000 | 871.577 | 0.878 | 5.357 | 0.706 | 378.348 |
| timesfm_2p5 | electricity | holdout | Europe | 21.000 | 1062.913 | 0.832 | 5.766 | 0.853 | 399.276 |
| chronos_2 | electricity | holdout | Europe | 21.000 | 1075.894 | 0.791 | 5.795 | 0.821 | 402.248 |
| sarima_core | electricity | holdout | Europe | 21.000 | 1114.908 | 1.026 | 6.186 | 0.794 | 426.210 |
| sarima_dcf | electricity | holdout | Europe | 17.000 | 1188.360 | 0.917 | 6.971 | 0.853 | 424.581 |
| nexus_llm | electricity | holdout | Europe | 21.000 | 1272.585 | 1.244 | 7.515 |  |  |
| xlstm | electricity | holdout | Europe | 21.000 | 1390.336 | 1.069 | 7.409 |  |  |
| seasonal_naive | electricity | holdout | India | 9.000 | 794.566 | 0.729 | 6.371 | 0.759 | 422.876 |
| timesfm_2p5 | electricity | holdout | India | 9.000 | 937.606 | 0.689 | 6.809 | 0.889 | 461.486 |
| chronos_2 | electricity | holdout | India | 9.000 | 1035.873 | 0.707 | 7.436 | 0.861 | 468.785 |
| sarima_core | electricity | holdout | India | 9.000 | 1050.009 | 0.748 | 7.769 | 0.843 | 507.321 |
| sarima_dcf | electricity | holdout | India | 9.000 | 1055.421 | 0.775 | 7.739 | 0.833 | 516.540 |
| nexus_llm | electricity | holdout | India | 9.000 | 1077.061 | 0.947 | 8.208 |  |  |
| xlstm | electricity | holdout | India | 9.000 | 1133.828 | 0.917 | 8.863 |  |  |
| chronos_2 | electricity | holdout | US | 36.000 | 687.763 | 0.895 | 5.171 | 0.736 | 355.145 |
| sarima_core | electricity | holdout | US | 36.000 | 712.978 | 0.971 | 5.442 | 0.801 | 372.460 |
| sarima_dcf | electricity | holdout | US | 21.000 | 717.537 | 0.990 | 6.481 | 0.770 | 392.077 |
| timesfm_2p5 | electricity | holdout | US | 36.000 | 776.180 | 0.954 | 5.638 | 0.778 | 383.025 |
| seasonal_naive | electricity | holdout | US | 36.000 | 798.397 | 1.027 | 5.860 | 0.720 | 400.113 |
| nexus_llm | electricity | holdout | US | 36.000 | 809.894 | 1.032 | 5.939 |  |  |
| xlstm | electricity | holdout | US | 36.000 | 843.871 | 1.103 | 6.322 |  |  |

## electricity, rolling

| model | resource | window | region | n_series | RMSE | MASE | WMAPE_pct | PICP | CRPS |
|---|---|---|---|---|---|---|---|---|---|
| sarima_dcf | electricity | rolling | All | 57.000 | 944.590 | 0.907 | 6.291 | 0.812 | 418.837 |
| chronos_2 | electricity | rolling | All | 66.000 | 954.015 | 0.920 | 6.059 | 0.762 | 405.426 |
| timesfm_2p5 | electricity | rolling | All | 66.000 | 967.127 | 0.938 | 6.224 | 0.784 | 411.203 |
| seasonal_naive | electricity | rolling | All | 66.000 | 1104.894 | 1.075 | 7.022 | 0.712 | 471.110 |
| sarima_core | electricity | rolling | All | 66.000 | 1112.172 | 1.034 | 6.863 | 0.770 | 458.006 |
| xlstm | electricity | rolling | All | 66.000 | 1333.640 | 1.185 | 8.130 |  |  |
| sarima_dcf | electricity | rolling | Europe | 20.000 | 1128.512 | 0.971 | 6.483 | 0.787 | 454.380 |
| timesfm_2p5 | electricity | rolling | Europe | 21.000 | 1153.143 | 0.973 | 6.392 | 0.774 | 426.480 |
| chronos_2 | electricity | rolling | Europe | 21.000 | 1155.783 | 0.959 | 6.236 | 0.765 | 427.937 |
| seasonal_naive | electricity | rolling | Europe | 21.000 | 1353.842 | 1.113 | 7.380 | 0.685 | 502.357 |
| sarima_core | electricity | rolling | Europe | 21.000 | 1448.754 | 1.073 | 7.661 | 0.747 | 518.428 |
| xlstm | electricity | rolling | Europe | 21.000 | 1707.914 | 1.298 | 9.044 |  |  |
| sarima_dcf | electricity | rolling | India | 9.000 | 1079.489 | 0.686 | 7.151 | 0.887 | 488.091 |
| timesfm_2p5 | electricity | rolling | India | 9.000 | 1263.182 | 0.830 | 9.259 | 0.806 | 577.898 |
| chronos_2 | electricity | rolling | India | 9.000 | 1278.089 | 0.843 | 9.268 | 0.833 | 586.562 |
| seasonal_naive | electricity | rolling | India | 9.000 | 1386.666 | 0.896 | 9.747 | 0.708 | 612.495 |
| sarima_core | electricity | rolling | India | 9.000 | 1410.479 | 0.838 | 9.077 | 0.843 | 586.231 |
| xlstm | electricity | rolling | India | 9.000 | 1666.885 | 1.032 | 12.073 |  |  |
| chronos_2 | electricity | rolling | US | 36.000 | 735.707 | 0.909 | 5.459 | 0.748 | 362.106 |
| sarima_dcf | electricity | rolling | US | 28.000 | 754.938 | 0.918 | 5.960 | 0.810 | 378.154 |
| timesfm_2p5 | electricity | rolling | US | 36.000 | 771.503 | 0.935 | 5.656 | 0.786 | 374.510 |
| sarima_core | electricity | rolling | US | 36.000 | 780.188 | 1.044 | 6.048 | 0.772 | 401.389 |
| seasonal_naive | electricity | rolling | US | 36.000 | 864.133 | 1.082 | 6.388 | 0.728 | 429.318 |
| xlstm | electricity | rolling | US | 36.000 | 971.401 | 1.145 | 6.979 |  |  |

## Check against the earlier cached benchmark (holdout)

| cache_model | fresh_model | n_points | median_rel_diff | p95_rel_diff | share_within_1pct | mean_zone_rmse_cache | mean_zone_rmse_fresh |
|---|---|---|---|---|---|---|---|
| SARIMA | sarima_core | 2280 | 0.000 | 0.000 | 1.000 | 322.027 | 322.027 |
| TimesFM | timesfm_2p5 | 2280 | 0.000 | 0.000 | 1.000 | 307.570 | 307.570 |
| Nexus | chronos_2 | 2280 | 0.000 | 0.000 | 1.000 | 296.369 | 296.369 |
| xLSTM | xlstm | 2280 | 0.011 | 0.099 | 0.463 | 393.816 | 381.520 |

## Failures

| model | resource | n |
|---|---|---|
| sarima_dcf | carbon | 3 |
| sarima_dcf | electricity | 101 |
