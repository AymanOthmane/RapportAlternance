from decimal import *
import pandas as pd
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import numpy as np
import QuantLib as ql
from datetime import datetime

"""This module contains a simplified version of the syndicated bond pricing class aimed at assisting Bond syndication team
    managed by Laurent Lagorsse in their day to day pricings.
    Author : Ayman Othmane
    """


class Bond:
    """
    Definition
    ----------
    This class contains a simplified version of the syndicated bond pricing class aimed at assisting Bond syndication team
    managed by Laurent Lagorsse in their day to day pricings.
    Author : Ayman Othmane
    """
    def __init__(self,
                 Start_Date, 
                 End_Date,
                 Jouissance,
                 First_Cpn_Date, 
                 yld, 
                 coupon, 
                 steps_per_year, 
                 convention = 'Act/Act',
                 IrrCpn = None, 
                 LastCpnDate=None,
                 Notional = 100
                 ):
    
        self.Start_Date =Start_Date
        self.End_Date=End_Date
        self.Jouissance=Jouissance
        self.First_Cpn_Date= First_Cpn_Date
        self.yld =yld
        self.coupon=coupon
        self.steps_per_year=steps_per_year
        self.convention = convention  
        self.IrrCpn = IrrCpn 
        self.LastCpnDate= LastCpnDate 
        self.Notional = Notional
        

    #fn_Long_Short_First calculates the time step and first cash flow in the case of an irregular first coupon
    def fn_Long_Short_First(self):
        Step = Decimal(self.date_difference(self.Start_Date,self.First_Cpn_Date,self.convention)/self.date_difference(self.Jouissance, self.First_Cpn_Date, self.convention))
        Price = (Decimal(self.coupon)*Step)/((1+Decimal(self.yld)) ** Step)

        return Price, Step

    #fn_Long_Short_Last calculates the time step and last cash flow in the case of an irregular last coupon
    def fn_Long_Short_Last(self, Price, Step):
        Last_Step = self.date_difference(self.LastCpnDate, self.End_Date, self.convention)/self.date_difference(self.End_Date.addyears(-1/self.steps_per_year),self.End_Date,self.convention)
        Step+=Last_Step
        Price += (1 + Decimal(self.coupon*Last_Step)) / ((1 + Decimal(self.yld)) ** Step)
        return Price
    

    def fn_Bond_Pricing(self):
        #Convertion of date strings into datetime objects
        try:
            self.Start_Date = datetime.strptime(self.Start_Date, '%d/%m/%Y').date()
            self.End_Date = datetime.strptime(self.End_Date, '%d/%m/%Y').date()
            self.Jouissance = datetime.strptime(self.Jouissance, '%d/%m/%Y').date()
            self.First_Coupon = datetime.strptime(self.First_Coupon, '%d/%m/%Y').date()
        except:
            pass

        #Definition of number of steps per year and number of years until maturity
        Maturity_in_years = self.date_difference(self.Start_Date, self.End_Date, self.convention)

        #Calculation of coupon and yield per periode
        self.coupon = self.coupon/self.steps_per_year
        self.yld = self.yld/self.steps_per_year

        #Setting initial values for price and step
        Price = 0
        Step=0

        #Setting initial step to step per year or calculation
        # in case of first irregular coupon and conversion to decimal object
        if self.IrrCpn == None or self.IrrCpn == "" or self.IrrCpn == "Last" :
            Step = 1
        else:   
            Price, Step = self.fn_Long_Short_First()
            Step+=1

        # Defining number of steps until maturity
        Steps = Maturity_in_years*self.steps_per_year
        
        # Loop through cash flows of bond and calculation of
        # actualised sum of cash flows at each step before maturity
        while Step < Steps:
            Price += Decimal(self.coupon)/((1+Decimal(self.yld)) ** Step)
            Step += 1


        #Calculating last cash flow in case of irregular last coupon
        if self.IrrCpn == "Last":
            Price=self.fn_Long_Short_Last(Price, Step, self.End_Date, self.LastCpnDate, self.coupon, self.yld)
        else:
            Price += Decimal(1+Decimal(self.coupon))/((1+Decimal(self.yld)) ** Decimal(Steps))

        #Converting Price from decimal object to float 
        Price = float(Price)

        #returning price for face value 100
        return Price*self.Notional


    # Convert Python datetime to QuantLib Date
    def to_quantlib_date(self, py_date):
        return ql.Date(py_date.day, py_date.month, py_date.year)

    # Function to calculate the date difference using different day count conventions
    def date_difference(self, start_date, end_date, convention):
        # Convert the dates to QuantLib dates
        ql_start_date = self.to_quantlib_date(start_date)
        ql_end_date = self.to_quantlib_date(end_date)
        
        # Define the appropriate day count convention
        if convention == 'Act/Act':
            day_count = ql.ActualActual(ql.ActualActual.Bond)
        elif convention == '30/360':
            day_count = ql.Thirty360(ql.ActualActual.Bond)
        elif convention == 'Act/360':
            day_count = ql.Actual360()
        elif convention == 'Act/365':
            day_count = ql.Actual365Fixed()
        else:
            raise ValueError(f"Unsupported day count convention: {self.convention}")
            
        
        # Calculate the year fraction (which represents the date difference)
        year_fraction = day_count.yearFraction(ql_start_date, ql_end_date)
        
        return year_fraction



def get_bond_template(template):
    if template == '5 ans - Annuel - T+2':
        return {
            "startDate" : datetime.today() + relativedelta(days=2),
            "endDate" : datetime.today()+ relativedelta(years=5, days=2), 
            "jouissance" :  datetime.today()+ relativedelta(days=2), 
            "first_coupon" : datetime.today()+ relativedelta(years=1, days = 2), 
            "frequency" : 1, 
        }
    elif template == '10 ans - Annuel - T+2':
        return {
            "startDate" : datetime.today()+ relativedelta(days=2),
            "endDate" : datetime.today()+ relativedelta(years=10, days = 2), 
            "jouissance" :  datetime.today()+ relativedelta(days=2), 
            "first_coupon" : datetime.today()+ relativedelta(years=1, days = 2), 
            "frequency" : 1, 
        }
    elif template =='5 ans - Semestriel - T+2':
        return {
            "startDate" : datetime.today()+ relativedelta(days=2), 
            "endDate" : datetime.today()+ relativedelta(years=5, days = 2), 
            "jouissance" :  datetime.today()+ relativedelta(days=2), 
            "first_coupon" : datetime.today()+ relativedelta(months=6, days = 2), 
            "frequency" : 2, 
        }
    elif template == '10 ans - Semestriel - T+2':
        return {
            "startDate" : datetime.today()+ relativedelta(days=2), 
            "endDate" : datetime.today()+ relativedelta(years=10, days = 2), 
            "jouissance" :  datetime.today()+ relativedelta(days=2), 
            "first_coupon" : datetime.today()+ relativedelta(months=6, days = 2), 
            "frequency" : 2, 
        }
    else:   
        return {
            "startDate" : datetime.today()+ relativedelta(days=2), 
            "endDate" : datetime.today()+ relativedelta(years=1, days=2), 
            "jouissance" :  datetime.today()+ relativedelta(days=2), 
            "first_coupon" : datetime.today()+ relativedelta(years=1, days = 2), 
            "frequency" : 1,
                }