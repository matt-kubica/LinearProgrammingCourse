from maxflow_example import run_example
import os

test_networks = ['04_05_ahuja.dnf', '06_08_wikipedia.dnf', '06_09_cormen.dnf', '06_10_europe.dnf', '06_10_jungnickel.dnf', '07_11_wikipedia.dnf']
test_dir = 'tests/maxflow/networks'
print("Running tests...")
print('=========')

success = True
for test_network in test_networks:
    test = os.path.join(test_dir, test_network)
    print(f'{test_network}', flush=True)
    try:
        run_example(test)
        print(f'** PASSED **', flush=True)
    except Exception as e:
        success = False
        print(f'** FAILED ** (message: {e})', flush=True)
    print('--------', flush=True)

print('=========')
if success:
    print("Congratulations, your maxflow algorithms seem to work correctly!")
else:
    print("Some of the tests failed. Fix your implementation ASAP :)")

    