"""
Uses the fast marching method to locate seismic events in 2d to explore effects
of network geometry.

Geometry is:
    [0, 1000]
    [0, 1000]

"""
import skfmm
import pandas as pd
import numpy as np
import xarray as xr

import matplotlib.pyplot as plt


def unravel_index(velmod: xr.DataArray, index,) -> np.ndarray:
    """Unravel the index to get a tuple. """
    # the index has already been transformed
    if isinstance(index, (tuple, np.ndarray)) and len(index) == len(velmod.shape):
        return index
    # else unravel
    return np.unravel_index(index, velmod.shape)


def make_tt_grids(df, velocity=1000, extents=(500, 500)):
    """Make travel time grids for each station"""

    def _get_phi(row, x_coords, y_coords):
        """get the phi grid."""
        x_ind = np.digitize(row["x"], x_coords)
        y_ind = np.digitize(row["y"], y_coords)
        if x_ind == extents[0]:
            x_ind = extents[0] - 1
        if y_ind == extents[1]:
            y_ind = extents[1] - 1
        phi = np.ones_like(velocity_grid)
        phi[x_ind, y_ind] = 0
        return phi

    out = {}
    x_coords = np.arange(extents[0])
    y_coords = np.arange(extents[1])
    velocity_grid = np.ones(extents) * velocity
    for ind, row in df.iterrows():
        coords = {"x": x_coords, "y": y_coords}
        phi = _get_phi(row, x_coords, y_coords)
        tt = skfmm.travel_time(phi=phi, speed=velocity_grid)
        dar = xr.DataArray(tt, dims=list(coords), coords=coords)
        out[ind] = dar

    ds = xr.Dataset(out)
    return ds.to_array(dim="station")


def make_picks(station_df, event_df, velocity, noise_std, num=1000):
    """
    Make num picks for each station/event pair adding noise.

    Parameters
    ----------
    station_df
        A dataframe of stations (must have cols x, y)
    event_df
        A dataframe of events (must have cols x, y)
    velocity
        The velocity of the media
    noise_std
        The std of the noise, centered on the pick
    num
        The number of time to relocate each event.
    """

    def _get_pre_noise_df(eid, eser):
        """get a pre-noise dataframe"""
        x_diff = np.abs(eser["x"] - station_df["x"])
        y_diff = np.abs(eser["y"] - station_df["y"])
        total_dist = np.sqrt(x_diff ** 2 + y_diff ** 2)
        out = (total_dist / velocity).to_frame(name="time")
        out["event"] = eid
        out["station"] = station_df.index
        out = pd.concat([out] * num, axis=0)
        # add iteration number
        num_ind = np.repeat(np.arange(num), len(station_df))
        out["iteration"] = num_ind
        return out.reset_index()

    def _get_noise(df) -> pd.DataFrame:
        """Get noise df to add to time"""
        rands = np.random.randn(len(df)) * noise_std
        df["time"] += rands
        return df

    out = []
    for eid, eser in event_df.iterrows():
        df = _get_pre_noise_df(eid, eser)
        out.append(_get_noise(df))

    return pd.concat(out, axis=0).reset_index(drop=True)


def locate_events(tt_grids, picks):
    """Locate events using traveltime grids and picks."""

    def _make_loc_ser(df, tt_grids, event_iteration):
        """pass"""
        dar = df[["time", "station"]].set_index("station").to_xarray()
        opt_grid = (tt_grids - dar).std(dim="station").to_array().squeeze()
        argmin = np.nanargmin(opt_grid.values)
        loc_ind = unravel_index(opt_grid, argmin)
        out = {
            "x": float(tt_grids.coords["x"][loc_ind[0]]),
            "y": float(tt_grids.coords["y"][loc_ind[1]]),
            "event": event_iteration[0],
            "iteration": event_iteration[1],
        }
        return pd.Series(out)

    out = []
    for sub, df in picks.groupby(["event", "iteration"]):
        out.append(_make_loc_ser(df, tt_grids, sub))
    return pd.DataFrame(out)


def make_plot(sta_df, eve_df, loc_df, extents):
    """Make a plot of stations and events."""
    fig = plt.Figure(figsize=(10, 10))
    # plot stations
    plt.plot(sta_df["x"], sta_df["y"], "^", ms=10)
    # plt.plot(eve_df['x'], eve_df['y'], '.')
    buff = extents[0] * 0.1, extents[1] * 0.1
    plt.xlim(0 - buff[0], extents[0] + buff[0])
    plt.ylim(0 - buff[1], extents[1] + buff[1])

    plt.plot(loc_df["x"], loc_df["y"], ".", alpha=0.1, color="0.5", ms=8)
    plt.xticks([])
    plt.yticks([])
    return fig


if __name__ == "__main__":
    noise_std = 0.005
    velocity = 1000
    num_iterations = 1000
    extents = (500, 500)
    # create station
    sta_df = pd.DataFrame(
        [
            [extents[0] / 2, 0],
            [0, 0],
            [0, extents[1] / 2],
            [extents[0] / 2, extents[1] / 2],
            # [extents[0] / 2, extents[1]],
            # [extents[0], extents[1]],
            # [extents[0], extents[1] / 2],
        ],
        columns=["x", "y"],
    )
    # create event list
    event_df = pd.DataFrame(
        [
            [extents[0] / 4, extents[1] / 4],
            [extents[0] * (2.5 / 4), extents[1] * (3 / 4)],
        ],
        columns=["x", "y"],
    )
    # get  travel time grids
    tt_grids = make_tt_grids(sta_df, velocity=velocity)
    picks = make_picks(
        sta_df, event_df, velocity=velocity, noise_std=noise_std, num=num_iterations
    )
    locs = locate_events(tt_grids, picks)
    # plot
    fig = make_plot(sta_df, event_df, locs, extents)
    plt.savefig("network_and_such.png", transparent=True, dpi=450)

    breakpoint()
