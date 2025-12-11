import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import healpy as hp
import healpy.projaxes as hp_projaxes
import matplotlib.pyplot as plt

from fwd_model_tools.fields import DensityField, DensityStatus, FieldStatus, FlatDensity, SphericalDensity


def _base_density_field():
    return DensityField(
        array=jnp.zeros((4, 4, 4)),
        mesh_size=(4, 4, 4),
        box_size=(100.0, 100.0, 100.0),
        observer_position=(0.5, 0.5, 0.5),
        nside=2,
        flatsky_npix=(4, 4),
        halo_size=0,
        status=FieldStatus.DENSITY_FIELD,
    )


def test_flatdensity_plot_with_ax_reuses_supplied_axes():
    density_field = _base_density_field()
    flat = FlatDensity.FromDensityMetadata(
        array=jnp.ones((4, 4)),
        density_field=density_field,
        status=DensityStatus.LIGHTCONE,
    )
    fig, axes = plt.subplots(1, 2, figsize=(6, 3))
    returned_fig, returned_axes = flat.plot(
        ax=axes[0],
        show_colorbar=False,
        show_ticks=False,
        titles=["Left"],
    )

    assert returned_fig is fig
    assert returned_axes.shape == (1, 1)
    assert axes[0].images, "imshow should attach image to provided axes"

    plt.close(fig)


def test_sphericaldensity_plot_proxy_preserves_axes_reference():
    density_field = _base_density_field()
    nside = density_field.nside
    npix = hp.nside2npix(nside)
    spherical = SphericalDensity.FromDensityMetadata(
        array=jnp.ones((npix, )),
        density_field=density_field,
        status=DensityStatus.LIGHTCONE,
    )
    fig, axes = plt.subplots(1, 2, figsize=(6, 3))
    spherical.plot(ax=axes[0], show_colorbar=False, titles=[""], apply_log=False)

    delegate = getattr(axes[0], "_healpy_delegate", None)
    assert delegate is not None, "axes should proxy to healpy Mollweide axes"
    assert isinstance(delegate, hp_projaxes.HpxMollweideAxes)

    axes[0].set_title("Custom title")
    assert delegate.get_title() == "Custom title"

    plt.close(fig)


def test_sphericaldensity_plot_handles_multiple_axes():
    density_field = _base_density_field()
    nside = density_field.nside
    npix = hp.nside2npix(nside)
    spherical = SphericalDensity.FromDensityMetadata(
        array=jnp.ones((npix, )),
        density_field=density_field,
        status=DensityStatus.LIGHTCONE,
    )
    spherical_hi = spherical

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    spherical.plot(ax=axes[0], show_colorbar=False, titles=["left"], apply_log=False)
    spherical_hi.plot(ax=axes[1], show_colorbar=False, titles=["right"], apply_log=False)

    assert isinstance(getattr(axes[0], "_healpy_delegate"), hp_projaxes.HpxMollweideAxes)
    assert isinstance(getattr(axes[1], "_healpy_delegate"), hp_projaxes.HpxMollweideAxes)

    axes[0].set_title("Left")
    axes[1].set_title("Right")
    assert getattr(axes[0], "_healpy_delegate").get_title() == "Left"
    assert getattr(axes[1], "_healpy_delegate").get_title() == "Right"

    plt.close(fig)
