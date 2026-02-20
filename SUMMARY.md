# Do UFO Sightings Cluster Near Underwater Canyons?

**A plain-language summary for non-scientists**

---

## The question

The ocean floor is not flat. In some places, massive underwater canyons cut into the continental shelf — some rivaling the Grand Canyon in scale. We asked: do UFO sightings happen more often near these features?

## What we did

We took 80,000+ UFO sighting reports from the National UFO Reporting Center (NUFORC, 1990-2014) and mapped them against detailed ocean floor data from NOAA. We controlled for the obvious explanations — population density (more people = more reports), military bases, ports, and Navy operating areas. We used standard statistical methods and made all code and data publicly available.

## What we found

Along the US West Coast, UFO sighting rates are significantly elevated near steep underwater canyons, even after accounting for population and military activity. The strongest effect appears in two regions:

- **Puget Sound** (Washington state): 6.8x more reports near canyon features than expected
- **San Diego**: 9.8x more reports near the Scripps/La Jolla canyon system

The statistical association (Spearman rho = 0.37, p = 0.0001) held up across every test we ran: different time periods, different canyon definitions, held-out data from 2014-2023, and multiple confound checks.

## What we did NOT find

The effect **does not appear** on the US East Coast. But this is likely a data limitation, not evidence against the hypothesis — East Coast canyons sit 100-400 km offshore (beyond the continental shelf edge), too far for land-based observers to report anything unusual above them. West Coast canyons, by contrast, come within 50 km of shore.

## What this means

This is a correlation, not an explanation. We do not know *why* sightings cluster near underwater canyons. Possible interpretations range from atmospheric effects caused by deep-water upwelling, to observer biases we haven't identified, to something genuinely anomalous. The finding is robust enough to warrant independent replication and further investigation.

## Key numbers

| Metric | Value |
|--------|-------|
| Total reports analyzed | 80,332 |
| West Coast grid cells tested | 102 |
| Primary correlation | rho = 0.37, p = 0.0001 |
| Out-of-sample replication (post-2014 data) | rho = 0.35, p = 0.0001 |
| Confounds tested and survived | population, military, ports, ocean depth, magnetic anomaly, coastal upwelling, Navy operating areas |

## Limitations

- The effect is regional (West Coast only), not global
- It concentrates in Puget Sound and San Diego — the rest of the West Coast shows weak signal
- NUFORC is a self-reported database with inherent biases
- This is a single-author analysis awaiting peer review
- No causal mechanism is proposed or implied

## How to verify

Everything is open-source. Clone the repository, install Python dependencies, download one bathymetry file from NOAA, and run two scripts. The headline result reproduces in under 5 minutes. See the [Quick Start](README.md#quick-start) section in the README.

---

**Author:** Antoni Wedzikowski — independent researcher, lawyer and legaltech founder, Warsaw, Poland
**Repository:** [github.com/antoniwedzikowski-rgb/uap-canyon-analysis](https://github.com/antoniwedzikowski-rgb/uap-canyon-analysis)
**Contact:** [LinkedIn](https://www.linkedin.com/in/antekwedzikowski/) · GitHub Issues
