from TestDriver import TestDriver


def main():
    ptas = TestDriver()
    ptas.testPTAS()

    dp = TestDriver()
    dp.testDP()

    heuristic = TestDriver()
    heuristic.TwoDKEDA()


if __name__ == "__main__":
    main()
