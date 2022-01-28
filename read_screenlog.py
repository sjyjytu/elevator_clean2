with open('screenlog.0', 'r') as f:
    with open('screenlog_part.0', 'w') as sf:
        i = 0
        for l in f:
            sf.write(l)
            sf.write('\n')
            i+=1
            if i > 200000:
                break