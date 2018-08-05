import pandas as pd
import matplotlib.pyplot as plt

def test_run(company):
    df = pd.read_csv('data/{}.csv'.format(company))
    plt.show()

if __name__ == "__main__":
    test_run("IBM")