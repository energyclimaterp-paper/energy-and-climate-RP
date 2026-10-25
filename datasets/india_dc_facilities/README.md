# India datacenter facility list (v0)

An open list of Indian datacenter facilities, compiled for this study from operator disclosures, regulatory
filings and market reports. It is the facility spine of the India account.

| File | Content |
|---|---|
| `India_DC_Facilities_v0.xlsx` | sheet `Facilities` (194 rows), plus `Summary` (coverage) and `Notes & Method` (deduplication rules and caveats) |
| `india_dc_facilities.psv` | the `Facilities` sheet as pipe-separated text |

## Columns

| Column | Meaning |
|---|---|
| `facility_name`, `operator`, `operator_family` | facility and operator names |
| `city`, `state` | location (coordinates are assigned in the pipeline from city centroids) |
| `cea_grid_region` | Indian regional grid (Northern, Western, Southern, Eastern, North-Eastern) |
| `capacity_mw`, `capacity_basis`, `op_capacity_mw` | stated capacity, what it refers to, and the operational capacity used |
| `status` | Operational, Under-construction, Announced or Uncertain |
| `area_sqft` | floor area where disclosed |
| `confidence` | High / Med / Low, the reliability of the capacity figure |
| `source_url`, `notes` | where each row comes from |

## How the account uses it

Of the 194 rows, 129 are operational. Ten hyperscale cloud regions are excluded (capacity not disclosed at the
region level), 30 operational rows have no cited capacity, and 18 could not be placed in a basin. That leaves
**71 colocation facilities with 962 MW** in the account (paper Appendix A).

## Licence

CC BY 4.0 ([`LICENSE-DATA.md`](../../LICENSE-DATA.md)): free to share and adapt with credit to the paper. The
pages linked in `source_url` remain under their owners' terms.
