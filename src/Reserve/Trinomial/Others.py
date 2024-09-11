import numpy as np
import scipy.stats as si
import matplotlib.pyplot as plt
import networkx as nx
import sys
import QuantLib as ql

sys.setrecursionlimit(100000)

def black_scholes_option_price(option_type, strike, maturity, vol, r, dividends_euro, spot_price):
    dividend_yield = dividends_euro / spot_price

    d1 = (np.log(spot_price / strike) + (r - dividend_yield + 0.5 * vol**2) * maturity) / (vol * np.sqrt(maturity))
    d2 = d1 - vol * np.sqrt(maturity)

    if option_type == 'Call':
        option_price = spot_price * np.exp(-dividend_yield * maturity) * si.norm.cdf(d1, 0.0, 1.0) - strike * np.exp(-r * maturity) * si.norm.cdf(d2, 0.0, 1.0)
        #delta = np.exp(-dividend_yield * maturity) * si.norm.cdf(d1, 0.0, 1.0)
        #vega = spot_price * np.exp(-dividend_yield * maturity) * np.sqrt(maturity) * si.norm.pdf(d1, 0.0, 1.0)/100
        #gamma = np.exp(-dividend_yield * maturity) * si.norm.pdf(d1, 0.0, 1.0) / (spot_price * vol * np.sqrt(maturity))
        # theta = df = np.exp(-r * maturity)
        #dfq = np.exp(-dividend_yield * maturity)
        #tmptheta = (1.0/365.0) * (-0.5 * spot_price * dfq * si.norm.pdf(d1) * vol / (np.sqrt(maturity)) - r * strike * df * si.norm.cdf(d2))
        #theta = dfq * tmptheta
        #rho = maturity * strike * np.exp(-r * maturity) * si.norm.cdf(d2, 0.0, 1.0)/100

    elif option_type == 'Put':
        option_price = strike * np.exp(-r * maturity) * si.norm.cdf(-d2, 0.0, 1.0) - spot_price * np.exp(-dividend_yield * maturity) * si.norm.cdf(-d1, 0.0, 1.0)
        #delta = -np.exp(-dividend_yield * maturity) * si.norm.cdf(-d1, 0.0, 1.0)
        #vega = spot_price * np.exp(-dividend_yield * maturity) * np.sqrt(maturity) * si.norm.pdf(d1, 0.0, 1.0) / 100
        #gamma = np.exp(-dividend_yield * maturity) * si.norm.pdf(d1, 0.0, 1.0) / (spot_price * vol * np.sqrt(maturity))
        #df = np.exp(-r * maturity)
        #dfq = np.exp(-dividend_yield * maturity)
        #tmptheta = (1.0 / 365.0) * (-0.5 * spot_price * dfq * si.norm.pdf(d1) * vol / (np.sqrt(maturity)) + r * strike * df * si.norm.cdf(-d2))
        #theta = dfq * tmptheta
        #rho = -maturity * strike * np.exp(-r * maturity) * si.norm.cdf(-d2, 0.0, 1.0)/100

    else:
        raise ValueError("Le type d'option doit être 'Call' ou 'Put'.")

    return option_price#, delta, vega, gamma,theta,rho

def plot_tree(self):
    G = nx.Graph()
    nodes = [(self.root, None, 0)]  # Utilisez une liste pour parcourir les nœuds de manière itérative

    while nodes:
        current, parent, depth = nodes.pop()
        G.add_node(current, pos=(current.spot, -depth))
        if parent is not None:
            G.add_edge(parent, current)

        # Ajouter les nœuds enfants à la liste pour traitement ultérieur
        if current.next_up:
            nodes.append((current.next_up, current, depth + 1))
        if current.next_mid:
            nodes.append((current.next_mid, current, depth + 1))
        if current.next_down:
            nodes.append((current.next_down, current, depth + 1))

    pos = nx.get_node_attributes(G, 'pos')
    labels = {node: f"{node.price:.3f}" for node in G.nodes()}
    #labels = {node: f"Spot: {node.spot:.2f}\nProba: {node.proba:.4f}" for node in G.nodes()}
    nx.draw(G, pos, labels=labels, with_labels=True, node_size=500, node_color='skyblue', font_size=8,
            font_color='black')
    plt.title("Arbre trinomial d'options")
    plt.axis('off')
    plt.show()

def convergence_BS(vol,rate,spot,div,pricing_date,matu,divExDate,strike,Type,Nat,Nb_step):
    differences = []
    range_step=range(2,Nb_step)
    for i in range_step:

       Nb_steps = i

       option = Option(strike, Type, Nat)
       tree = Tree(vol, rate, spot, div, divExDate, Nb_steps, matu, pricing_date)
       tree.Build_Tree()
       diff= (tree.root.compute_pricing_node(option) - black_scholes_option_price(Type, strike,tree.compute_date(matu, pricing_date),vol, rate, div, spot))*i
       differences.append(diff)
       del tree
    return plot_convergence(differences,range_step)


def plot_convergence(differences, range_step):
    plt.figure()
    plt.plot(range_step, differences)
    plt.xlabel('Nombre de pas (i)')
    plt.ylabel('Différence (Arbre - Black-Scholes)')
    plt.title('Convergence du prix de l option')
    plt.grid(True)
    plt.show()


def to_quantlib_date(py_date):
    return ql.Date(py_date.day, py_date.month, py_date.year)

# Function to calculate the date difference using different day count conventions
def date_difference(start_date, end_date):
    # Convert the dates to QuantLib dates
    ql_start_date = to_quantlib_date(start_date)
    ql_end_date = to_quantlib_date(end_date)

    day_count = ql.ActualActual(ql.ActualActual.Bond)

    year_fraction = day_count.yearFraction(ql_start_date, ql_end_date)
    
    return year_fraction

import numpy as np

def monte_carlo_pricer(S0, K, T, r, sigma, option_type='call', num_simulations=10000, num_steps=252):
    """
    Monte Carlo pricer for European options using Black-Scholes assumptions.

    Parameters:
    S0 : float
        Initial stock price (underlying asset price).
    K : float
        Strike price of the option.
    T : float
        Time to maturity in years (e.g., 1 year is T=1.0).
    r : float
        Risk-free interest rate (annualized).
    sigma : float
        Volatility of the underlying asset (annualized).
    option_type : str, optional
        'call' for a call option, 'put' for a put option (default is 'call').
    num_simulations : int, optional
        Number of Monte Carlo simulations (default is 10000).
    num_steps : int, optional
        Number of time steps in the simulation (default is 252 for daily steps in 1 year).

    Returns:
    float
        Estimated price of the option.
    """

    dt = T / num_steps  # Time step
    discount_factor = np.exp(-r * T)  # Discount factor for present value

    # Simulate underlying asset price paths
    price_paths = np.zeros((num_simulations, num_steps + 1))
    price_paths[:, 0] = S0

    for t in range(1, num_steps + 1):
        # Random component for the asset price path (normally distributed)
        Z = np.random.standard_normal(num_simulations)
        price_paths[:, t] = price_paths[:, t - 1] * np.exp((r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z)

    # Calculate payoff for each simulation at maturity
    if option_type == 'Call':
        payoffs = np.maximum(price_paths[:, -1] - K, 0)
    elif option_type == 'Put':
        payoffs = np.maximum(K - price_paths[:, -1], 0)
    else:
        raise ValueError("Invalid option_type. Choose 'call' or 'put'.")

    # Monte Carlo estimate of the option price
    option_price = discount_factor * np.mean(payoffs)

    return option_price


