from collections import OrderedDict


with open("./envs/dss_123/IEEE123Loads.DSS", 'r') as dssfile:
    dss_str = dssfile.readlines()

load_base = OrderedDict() # a dictionary with order
for s in dss_str:
    #print(s)  # each line is a list
    if 'New Load.' in s:
        idx = s.index("New Load.") + len('New Load.')
        #print(s.index("New Load."))
        name = []
        for c in s[idx:]:  # contain all sentences after New Load tii end :
            #print(c)
            #print(s[idx:])
            if c == ' ':  # get out after the fist space
                #print(c)
                break
            else:
                name.append(c)
        name = ''.join(name)  # join all items in a tuple into a string and use '' as separator.
        #print(name)

        idx_kW = s.index("kW=") + len('kW=')
        #print(s.index("kW="))
        kW = []
        for c in s[idx_kW:]:
            if c == ' ':
                break
            else:
                kW.append(c)
        kW = float(''.join(kW))

        idx_kvar = s.lower().index("kvar=") + len('kvar=')
        kvar = []
        for c in s[idx_kvar:]:
            if c == ' ':
                break
            else:
                kvar.append(c)
        kvar = float(''.join(kvar))

        load_base[name] = (kW, kvar)

print(load_base)

load_node = OrderedDict()
# create dictionary for phases
phases = {'1': 'a',
          '2': 'b',
          '3': 'c',
          ' ': ''}
for s in dss_str:
    if 'Bus1=' in s:
        print(s)
        idx = s.index("Bus1=") + len('Bus1=')  # s.index("Bus1=")=15
        #print(s.index("Bus1="))
        name = []  # again create an empty list for name of the load
        for c in s[idx:]:
            print(c)
            if c == '.':
                p = s[idx + len(name) + 1]
                print(p)
                break
            elif c == ' ':
                p = ' '
                break
            else:
                name.append(c)

        name = ''.join(name).lower()
        load_node[name] = phases[p]

print(load_node)


