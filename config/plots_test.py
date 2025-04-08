import pandas as pd
import matplotlib.pyplot as plt

# Load the data
tbkan = pd.read_csv('/home/nikki/kan/data/tbkan_data.csv')
tbnn = pd.read_csv('/home/nikki/kan/data/tbnn_data.csv')
dns = pd.read_csv('/home/nikki/kan/data/DNS_SD_2000.csv')
rans = pd.read_csv('/home/nikki/kan/data/rans_SD.csv')

# Filter for centerline along Z at x = 2.5, y = 0.0
x_target, y_target = 2.5, 0.0
tolerance = 1e-4

# Extract TBKAN and TBNN lines
tbkan_line = tbkan[(abs(tbkan['Points_0'] - x_target) < tolerance) &
                   (abs(tbkan['Points_1'] - y_target) < tolerance)]
tbnn_line = tbnn[(abs(tbnn['Points_0'] - x_target) < tolerance) &
                 (abs(tbnn['Points_1'] - y_target) < tolerance)]

# Use coordinates from RANS to filter for x ≈ 2.5 and y ≈ 0
rans_line = rans[(abs(rans['komegasst_C_1'] - x_target) < tolerance) &
                 (abs(rans['komegasst_C_2'] - y_target) < tolerance)]

# Use RANS coordinates to align with DNS
dns_line = dns.loc[rans_line.index]

# Extract Z and Ux from all
z_tbkan, ux_tbkan = tbkan_line['Points_2'], tbkan_line['U_0']
z_tbnn, ux_tbnn = tbnn_line['Points_2'], tbnn_line['U_0']
z_rans, ux_rans = rans_line['komegasst_C_3'], rans_line['komegasst_U_1']
z_dns, ux_dns = rans_line['komegasst_C_3'], dns_line['REF_U_1']  # aligned by index

# Sort all by Z
sort_idx = z_dns.argsort()
z_dns, ux_dns = z_dns.iloc[sort_idx], ux_dns.iloc[sort_idx]
z_rans, ux_rans = z_rans.iloc[sort_idx], ux_rans.iloc[sort_idx]
z_tbkan, ux_tbkan = z_tbkan.sort_values(), ux_tbkan.loc[z_tbkan.sort_values().index]
z_tbnn, ux_tbnn = z_tbnn.sort_values(), ux_tbnn.loc[z_tbnn.sort_values().index]

# Plot
plt.figure(figsize=(7.5, 5))
#plt.plot(z_dns, ux_dns, label='DNS $U_x$', color='black', linestyle='--')
#plt.plot(z_rans, ux_rans, label='RANS $U_x$', color='gray')
plt.plot(z_tbnn, ux_tbnn, label='TBNN $U_x$', color='steelblue')
plt.plot(z_tbkan, ux_tbkan, label='TBKAN $U_x$', color='tomato')

plt.xlabel(r'$y$')
plt.ylabel(r'$U_x$')
plt.title(r'Comparison of $U_x$')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig('/home/nikki/kan/data/Ux_profile_comparison_all.pdf', dpi=300)
plt.savefig('/home/nikki/kan/data/Ux_profile_comparison_all.jpeg', dpi=300)
plt.show()
