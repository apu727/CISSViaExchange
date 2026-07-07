import numpy as np

failCount = 0
def expect(val, found, string):
    if (val == found):
        return 0
    global failCount
    failCount+=1
    print(f"Test failed: {string}")
    print(f"expected: {val}")
    print(f"found: {found}")
    return 1

def expectNear(val, found, string):
    if (np.abs(val - found) < 1e-13*found):
        return 0
    global failCount
    failCount+=1
    print(f"Test failed: {string}")
    print(f"expected: {val}")
    print(f"found: {found}")
    print(f"Diff: {val-found}")
    return 1

def expectNear(val, found, tol, string):
    if (np.abs(val - found) < tol*found):
        return 0
    global failCount
    failCount+=1
    print(f"Test failed: {string}")
    print(f"expected: {val}")
    print(f"found: {found}")
    print(f"Diff: {val-found}")
    return 1

def expectMatrixNear(val, found, tol, string):
    if (np.all(np.abs(val - found) <= tol*np.abs(found)+tol ) or np.linalg.norm(val - found) < tol):
        return 0
    global failCount
    failCount+=1
    print(f"Test failed: {string}")
    print(f"expected: {val}")
    print(f"found: {found}")
    print(f"Diff: {val-found}")
    return 1