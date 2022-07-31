# from [0, 32] to [0.9, 1.1]
for tap in range(0, 33):
        pu_per_ltc_tap = 5 / 8 / 100  # 5/8 % voltage rule
        tap_pu = 1.0 + (tap - 16) * pu_per_ltc_tap
        print(tap, tap_pu)

# from [0.9, 1.1] to [0, 32]
for tap_pu/ in range(9, 12):
    pu_per_ltc_tap = 5 / 8 / 100
    tap = (tap_pu - 1.0) / pu_per_ltc_tap + 16
    print(tap_pu, tap)
