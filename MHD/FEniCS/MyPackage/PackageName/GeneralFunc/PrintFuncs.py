import scipy
import time
import sys
from numpy.linalg import norm

def LabelStucture( Values,Label):
    for x,y in zip(Values,Label):
        d.setdefault(y, []).append(x)
    return d

def PandasFormat(table, field, format):
    table[field] = table[field].map(lambda x: format %x)
    return table


def  PrintStr(string, indent, boarder, preLines="", postLines=""):
    AppendedString  = ""
    for i in range(indent):
        AppendedString = " "+AppendedString

    StringPrint = AppendedString+string
    if indent < 2:
        Outerboarder = ""
        if indent == 1:
            for i in range(len(string)+indent+1):
                Outerboarder += boarder
        else:
            for i in range(len(string)+indent):
                Outerboarder += boarder
    else:
        AppendedString  = ""
        for i in range(indent-2):
            AppendedString = " "+AppendedString
        Outerboarder = AppendedString
        for i in range(len(string)+4):

            Outerboarder += boarder
    print preLines + Outerboarder
    print StringPrint
    print Outerboarder + postLines


def StrTimePrint(String, EndTime):
    print ("{:40}").format(String), " ==>  ",("{:4f}").format(EndTime),  ("{:9}").format("   time: "), ("{:4}").format(time.strftime('%X %x %Z')[0:5])

def PrintDOF(A):
    PrintStr('Number of DoFs',4,'-','\n','\n')
    for i in range(len(A)):
        print ("{:15}").format('   '+A.keys()[i]),' DOF = ',str(int(A.values()[i].dim()))


def Error(string):
    boarder = ' '
    filler = ' |'
    for i in range(len(string)+6):
        boarder += '-'
        if i > 1 and i < len(string)+6:
            filler += ' '
    filler += '|'
    print '\n\n\n' + boarder + '\n' + filler + '\n |  ' + string + '  |\n' + filler +'\n' + boarder + '\n\n\n'

    sys.exit()

def NormPrint(V, Type):
    PrintStr('Non-linear errors',4,'-','\n','\n')
    if Type != 'Update':
        if len(V) == 2:
            print '  Velocity-Norm = ', V[0]
            return V[0]
        else:
            print '  Velocity-Norm = ', V[0]
            print '  Magnetic-Norm = ', V[2]
            return V[0] + V[2]
    else:
        if len(V) == 2:
            print '  Velocity-Norm = ', V[0]
            print '  Pressure-Norm = ', V[1]
            return V[0]+V[1]
        else:
            print '  Velocity-Norm =  ', V[0]
            print '  Pressure-Norm =  ', V[1]
            print '  Magentic-Norm =  ', V[2]
            print '  Multiplier-Norm = ', V[3]
            return V[0]+V[1]