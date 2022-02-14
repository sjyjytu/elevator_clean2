line = """47.18 66.32 113.50 1513634.46 for 147 people
50.33 68.09 118.42 1366967.83 for 140 people
52.68 63.27 115.95 1581352.13 for 138 people
73.59 64.84 138.43 1879697.26 for 147 people
40.48 64.78 105.26 1597301.07 for 128 people
33.65 67.72 101.37 1418403.45 for 133 people
53.16 68.84 122.01 1475248.87 for 155 people
49.31 74.20 123.51 1751453.08 for 121 people
72.65 70.93 143.58 1640769.05 for 158 people
58.80 58.34 117.14 1961945.72 for 140 people
35.82 68.45 104.26 1523808.22 for 138 people
66.73 70.60 137.33 1682556.17 for 141 people
53.22 74.82 128.04 1473873.34 for 147 people
67.30 68.37 135.66 1796283.54 for 146 people
57.62 64.41 122.03 1731933.29 for 153 people
55.60 59.13 114.73 1584139.56 for 146 people
51.05 68.72 119.77 1414607.95 for 140 people
57.37 62.80 120.17 1729697.28 for 147 people
42.08 72.14 114.22 1470602.99 for 156 people
45.19 64.84 110.03 1618287.90 for 134 people"""
line = line.split('\n')
AWT = 0
AST = 0
ENERGY = 0
i = 0
# data_idx = [0,2,7,12,13,15,16,17,19]
data_idx = range(20)
for l in line:
    if i in data_idx:
        l = l.split(' ')
        awt = l[0]
        ast = l[2]
        energy = l[3]
        AWT += float(awt)
        AST += float(ast)
        ENERGY += float(energy)
        print(i, awt, ast, energy)
    i += 1
print(f'AWT: {AWT / len(data_idx):.2f}, AWT: {AST / len(data_idx):.2f}, ENERGY: {ENERGY / len(data_idx):.0f}')
