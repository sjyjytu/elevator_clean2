line = """-------------------------------------------------evaluation result-------------------------------------------------
[Evaluation] for 114 people: mean waiting time 26.2, mean transmit time: 59.4, sum time: 85.6. Total energy: 1665798.7411959202.
-------------------------------------------------evaluation result-------------------------------------------------
[Evaluation] for 118 people: mean waiting time 36.4, mean transmit time: 56.1, sum time: 92.6. Total energy: 1896492.2422886216.
-------------------------------------------------evaluation result-------------------------------------------------
[Evaluation] for 116 people: mean waiting time 34.5, mean transmit time: 55.2, sum time: 89.7. Total energy: 1890394.9814032475.
-------------------------------------------------evaluation result-------------------------------------------------
[Evaluation] for 120 people: mean waiting time 32.8, mean transmit time: 54.0, sum time: 86.7. Total energy: 2027718.0799830991.
-------------------------------------------------evaluation result-------------------------------------------------
[Evaluation] for 112 people: mean waiting time 39.8, mean transmit time: 54.1, sum time: 93.8. Total energy: 2089697.5214160334.
-------------------------------------------------evaluation result-------------------------------------------------
[Evaluation] for 109 people: mean waiting time 36.5, mean transmit time: 58.0, sum time: 94.5. Total energy: 1948650.9723188495.
-------------------------------------------------evaluation result-------------------------------------------------
[Evaluation] for 124 people: mean waiting time 41.3, mean transmit time: 56.5, sum time: 97.8. Total energy: 1955598.1227800124.
-------------------------------------------------evaluation result-------------------------------------------------
[Evaluation] for 92 people: mean waiting time 28.0, mean transmit time: 55.5, sum time: 83.5. Total energy: 2264523.259779644.
-------------------------------------------------evaluation result-------------------------------------------------
[Evaluation] for 118 people: mean waiting time 35.7, mean transmit time: 57.6, sum time: 93.2. Total energy: 1809617.9522509938.
-------------------------------------------------evaluation result-------------------------------------------------
[Evaluation] for 118 people: mean waiting time 40.5, mean transmit time: 60.6, sum time: 101.1. Total energy: 1938840.3070426215.
-------------------------------------------------evaluation result-------------------------------------------------
[Evaluation] for 100 people: mean waiting time 26.6, mean transmit time: 51.8, sum time: 78.4. Total energy: 2207044.175384534.
-------------------------------------------------evaluation result-------------------------------------------------
[Evaluation] for 109 people: mean waiting time 30.1, mean transmit time: 62.3, sum time: 92.4. Total energy: 1677430.2958962831.
-------------------------------------------------evaluation result-------------------------------------------------
[Evaluation] for 116 people: mean waiting time 35.9, mean transmit time: 64.0, sum time: 99.9. Total energy: 1995607.0345602205.
-------------------------------------------------evaluation result-------------------------------------------------
[Evaluation] for 125 people: mean waiting time 42.6, mean transmit time: 59.7, sum time: 102.3. Total energy: 1864518.8363468423.
-------------------------------------------------evaluation result-------------------------------------------------
[Evaluation] for 128 people: mean waiting time 56.4, mean transmit time: 53.8, sum time: 110.2. Total energy: 2399553.9211477158.
-------------------------------------------------evaluation result-------------------------------------------------
[Evaluation] for 117 people: mean waiting time 36.4, mean transmit time: 50.7, sum time: 87.1. Total energy: 2197648.804371564.
-------------------------------------------------evaluation result-------------------------------------------------
[Evaluation] for 114 people: mean waiting time 25.8, mean transmit time: 58.5, sum time: 84.2. Total energy: 1672754.2519949758.
-------------------------------------------------evaluation result-------------------------------------------------
[Evaluation] for 114 people: mean waiting time 28.5, mean transmit time: 53.7, sum time: 82.2. Total energy: 1799523.6487741403.
-------------------------------------------------evaluation result-------------------------------------------------
[Evaluation] for 127 people: mean waiting time 48.8, mean transmit time: 58.1, sum time: 106.9. Total energy: 2017139.4295874287.
-------------------------------------------------evaluation result-------------------------------------------------
[Evaluation] for 113 people: mean waiting time 25.9, mean transmit time: 51.2, sum time: 77.1. Total energy: 1960156.576611793.
average awt: 35.43, average att: 56.54, average ast: 91.97, average energy: 1963935

Process finished with exit code 0"""
line = line.split('\n')
AWT = 0
AST = 0
ENERGY = 0
i = 0
# data_idx = [0,2,7,12,13,15,16,17,19]
data_idx = range(20)
for l in line:
    if l.startswith('[Evaluation]'):
        if i in data_idx:
            l = l.split(' ')
            print(l)
            awt = l[7][:-1]
            att = l[11][:-1]
            ast = l[14][:-1]
            energy = l[-1][:-1]
            AWT += float(awt)
            AST += float(ast)
            ENERGY += float(energy)
            print(i, awt, ast, energy)
        i += 1
print(f'AWT: {AWT / len(data_idx):.2f}, AWT: {AST / len(data_idx):.2f}, ENERGY: {ENERGY / len(data_idx):.0f}')