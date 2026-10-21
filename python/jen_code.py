class ESBLMarkovModel:
    def _antibiotic_rate(self, year, intervention):
        if not intervention:
            return self.p["antibiotic_rate"]
        start, end = self.p["intervention_start_ab"], self.p["intervention_end_ab"]
        if year <= 1:  return start
        if year >= 9:  return end
        return start + (end - start) * (year - 1) / 8.0  # linear interp years 1–9

    def run(self, n_years=10, intervention=False, population=100_000):
        C, NC = self.p["colonization_prev"], 1 - self.p["colonization_prev"]

        for year in range(1, n_years + 1):
            ab = self._antibiotic_rate(year, intervention)

            # NC → C: antibiotic-mediated + household (independent competing risks)
            # β × C: within-household probability × proportion of households with colonised member
            p_nc_to_c = 1 - (1 - ab * self.p["p_colonize_ab"]) \
                          * (1 - self.p["household_transmission"] * C)

            # C → NC: spontaneous decolonisation
            p_c_to_nc = self.p["decolonization_rate"]

            # Infections: NC + colonised without ab + colonised with ab
            infections = (NC          * self.p["p_infect_nc"]        +
                          C*(1 - ab)  * self.p["p_infect_col_no_ab"] +
                          C*ab        * self.p["p_infect_col_ab"]) * population

            # State update
            C, NC = NC*p_nc_to_c + C*(1 - p_c_to_nc), NC*(1 - p_nc_to_c) + C*p_c_to_nc