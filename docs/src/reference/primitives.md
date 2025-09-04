
# Parallel primitives

## Gather

```@docs
gather
gather!
allocate_gather
```

## Scatter

```@docs
scatter
scatter!
allocate_scatter
```

## Multicast

```@docs
multicast
multicast!
allocate_multicast
```

## Scan

```@docs
scan
```

## Reduction

```@docs
reduction
```

## Exchange

```@docs
ExchangeGraph
ExchangeGraph(snd,rcv)
ExchangeGraph(snd)
exchange
exchange!
allocate_exchange
default_find_rcv_ids
set_default_find_rcv_ids
find_rcv_ids_gather_scatter
find_rcv_ids_ibarrier
```
