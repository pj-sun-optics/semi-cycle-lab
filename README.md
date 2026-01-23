## Motivation
Treat the semiconductor market as a noisy dynamical system.
This project builds a reproducible weekly cycle dashboard with graceful degradation.

## Data
- Fund NAV (007301)
- CSIndex valuation (H30184)
- Manual macro signals (SIA / memory / capex / inventory)

## Method
- State variables → Cycle Score
- Decision rule: Continue DCA / Stop-add only
- No short-term trading, no auto sell

## Output
- Weekly Markdown reports
- 15-week rolling index


## TODO
- [ ] Valuation cache fallback
- [ ] Long-run score vs drawdown analysis
- [ ] Parameter sensitivity test
