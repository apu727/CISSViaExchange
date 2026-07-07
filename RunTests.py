import pyTests.Util
if __name__ == "__main__":
    import pyTests.NHCLFTest1 
    pyTests.NHCLFTest1.runHamGen()
    if pyTests.Util.failCount > 0:
        print("Test Failed")
        #quit(1)
        
    import pyTests.NHCLFTest2 
    pyTests.NHCLFTest2.runHamGen()
    if pyTests.Util.failCount > 0:
        print("Test Failed")
        #quit(1)

    import pyTests.NHCLFTest3 
    pyTests.NHCLFTest3.runHamGen()
    if pyTests.Util.failCount > 0:
        print("Test Failed")
        #quit(1)

    import pyTests.CU2Test1 
    pyTests.CU2Test1.runHamGen()

    if pyTests.Util.failCount > 0:
        print("Tests Failed, Please see log")
        quit(1)
    else:
        print("All Good!")
        quit(0)

