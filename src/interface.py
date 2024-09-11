import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
from dateutil.relativedelta import relativedelta
import QuantLib

from pricer_bond import *
from Reserve.Trinomial import Class_Objects as Tri, Others

st.set_page_config(layout="wide")
onglet = st.sidebar.radio("Navigation", ["Pricer Bond","Reserve Marché"])
template = get_bond_template('N/A')
# print(template)
if onglet == "Pricer Bond":
    st.title('Pricer Obligation')
    ColA, ColB = st.columns(2)

    with ColA:
        with st.form(key='form_2'):
            st.subheader("Parametres")
            submit_button_2 = st.form_submit_button(label= 'Price Bond', use_container_width=True)
            Col1, Col2 = st.columns(2)
            with Col1:
                notional = st.number_input("Notionnel",1000,10**9,step=1000,value=1000) 
                startDate = st.date_input("Date de pricing",format='DD/MM/YYYY', value=template['startDate'])
                endDate = st.date_input("Date de Maturité",format='DD/MM/YYYY',value=template['endDate'])
                jouissance =  st.date_input("Début de calcul du 1er coupon",format='DD/MM/YYYY',value=template['jouissance']) 
                first_coupon = st.date_input("Paiement du premier coupon",format='DD/MM/YYYY',value=template['first_coupon'])
                convention = st.selectbox("Convention", ['Act/Act','30/360','Act/360','Act/365'])

            with Col2:
                
                frequency = st.number_input("Fréquence annuel ",1,12,value=template['frequency'], help="nb de paiement par an, i.e. pour semestriel, 2")
                yld = st.number_input("Yield en %",0.00,100.00,5.00,0.01)/100
                cpn = st.number_input("Couponen %",0.00,100.00,5.00,0.01)/100
                irrg_cpn =st.selectbox("Coupon irrégulier? ", ['','First','Last'], help="Laisser vide si aucun")  
                LastCpnDate = st.date_input("Dernier coupon avant maturité ",format='DD/MM/YYYY',help='Valable seulement si coupon irrégulier = True')
                if irrg_cpn != 'Last': 
                    LastCpnDate = None

    with ColB:    
        if submit_button_2:
            bond = Bond(startDate,endDate,jouissance,first_coupon,yld,cpn,frequency,convention,irrg_cpn,LastCpnDate,notional)
            price = bond.fn_Bond_Pricing()
            st.subheader(f" Bond clean price is {price:4f}")
            # Notify the reader that the data was successfully loaded.
            st.text(f"Priced on  {datetime.now().strftime('%d/%m/%Y')}")



    st.markdown(""" 
            <style>
                div[data-testid="column"] {
                    width: fit-content !important;
                    flex: unset;
                }
                div[data-testid="column"] * {
                    width: fit-content !important;
                }
            </style>
            """, unsafe_allow_html=True)



elif onglet == "Reserve Marché":
# Create a text element and let the reader know the data is loading.
    st.title('Reserve Marché')
    colA, colB = st.columns(2)
    with colA:
        with st.form(key='form_5'):
            # source = st.selectbox("Source des données",["Bloomberg","CSV File"])
            st.subheader("Parametres")
            submit_button = st.form_submit_button(label= 'Marquer les Reserves', use_container_width=True)
            col1, col2 = st.columns(2)
            
            with col1:
                startDate = st.date_input("Date de pricing",format='DD/MM/YYYY', value=datetime.today() + relativedelta(days = 2))
                endDate = st.date_input("Date de Maturité",format='DD/MM/YYYY', value=datetime.today() + relativedelta(years=1, days = 2))
                exDate =  st.date_input("Ex-date de dividende",format='DD/MM/YYYY', value=datetime.today() + relativedelta(months=6, days = 2))
                # spot = st.number_input("Spot",value=100)
                # strike = st.number_input("Strike",value=100)
                div = st.number_input("dividende",value=0)
                Type=st.selectbox("Type d'option", ['Call','Put'])  
                Nat=st.selectbox("Nature d'option", ['Eu','Am'])   

            with col2:
                notional = st.number_input("Notionnel",1000,10**9,step=1000,value=1000)
                spot = st.number_input("Spot",value=100)
                strike = st.number_input("Strike",value=100)
                Volatility = st.number_input("Volatilité",0.0,1.0,value=0.2)
                riskfree = st.number_input("Taux sans risque",0.0,1.0,value=0.05)
                Nb_steps = st.number_input("Nomber de pas",100,10**5,value=10**3,step=100)
           

    with colB: 
        if submit_button:
            maturity = Others.date_difference(startDate,endDate)

            # Define the appropriate day count convention


            option =  Tri.Option(strike, Type,Nat)
            tree = Tri.Tree(Volatility, riskfree, spot, div, exDate, Nb_steps, endDate,startDate)
            tree.Build_Tree()
            Trinom_prix=tree.root.compute_pricing_node(option)
            BS_prix = Others.black_scholes_option_price(Type,strike,maturity,Volatility,riskfree,div,spot)
            MC_prix = Others.monte_carlo_pricer(spot,strike,maturity,riskfree,Volatility,Type,100000,int(252*maturity))
            st.subheader(f"Prices")
            st.subheader(f"Trinomial Tree price: {notional/spot * Trinom_prix:4f}")

            st.subheader(f"Black & Schole price: {notional/spot*BS_prix:4f}")

            st.subheader(f"Monte Carlo price: {notional/spot*MC_prix:4f}")
            col3, col4 = st.columns(2)

            with col3:
                R_cons = max(Trinom_prix, BS_prix, MC_prix) - min (Trinom_prix, BS_prix, MC_prix)
                st.subheader('Reserve Conservatrice')
                st.subheader(f"{notional/spot*R_cons: 4f}")

            with col4:
                R_moy = np.absolute(np.diff([Trinom_prix, BS_prix, MC_prix])).mean()
                st.subheader('Reserve Moyenne')
                st.subheader(f"{notional/spot* R_moy: 4f}")


