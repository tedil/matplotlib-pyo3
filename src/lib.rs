use anyhow::Result;
use ndarray;
use numpy::{PyArray1, ToPyArray};
pub use pyo3;
use pyo3::types::IntoPyDict;
use pyo3::Python;

pub trait PlotExt<'a> {
    fn plot(plt: &mut PyPlot<'a>) -> Result<()>;
}

/// Wrapper around some methods and classes of `matplotlib.pyplot`.
pub struct PyPlot<'a> {
    py: Python<'a>,
    plt: &'a pyo3::types::PyModule,
}

impl<'a> PyPlot<'a> {
    pub fn new(py: Python<'a>) -> Result<Self> {
        let plt = py.import("matplotlib.pyplot")?;
        Ok(Self { py, plt })
    }

    /// Create a new [Figure].
    /// See `matplotlib.pyplot.figure` for more details.
    pub fn figure(&self) -> Result<Figure> {
        let fig = self.plt.call0("figure")?;
        Ok(Figure { py: self.py, fig })
    }

    /// Get the current figure.
    ///
    /// If no current figure exists, a new one is created using figure().
    /// See also: https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.gcf.html
    pub fn gcf(&self) -> Result<Figure> {
        let fig = self.plt.call_method0("gcf")?;
        Ok(Figure { py: self.py, fig })
    }

    pub fn show(&self) -> Result<&'a pyo3::PyAny> {
        Ok(self.plt.call0("show")?)
    }
}

pub struct Figure<'a> {
    py: Python<'a>,
    fig: &'a pyo3::types::PyAny,
}

impl<'a> Figure<'a> {
    pub fn add_axes(
        &self,
        left: f64,
        bottom: f64,
        width: f64,
        height: f64,
        share_x: Option<&'a Axes<'a>>,
        share_y: Option<&'a Axes<'a>>,
    ) -> Result<Axes> {
        let args = PyArray1::from_vec(self.py, vec![left, bottom, width, height]);
        let mut shares = vec![];
        if let Some(ax) = share_x {
            shares.push(("sharex", ax.axes));
        }
        if let Some(ax) = share_y {
            shares.push(("sharey", ax.axes));
        }
        let axis = self
            .fig
            .call_method("add_axes", (args,), Some(shares.into_py_dict(self.py)))?;
        Ok(Axes {
            py: self.py,
            axes: axis,
        })
    }

    pub fn gca(&self) -> Result<Axes> {
        let axes = self.fig.call_method0("gca")?;
        Ok(Axes { py: self.py, axes })
    }

    pub fn show(&self) -> Result<&'a pyo3::PyAny> {
        Ok(self.fig.call_method0("show")?)
    }
}

pub struct Axes<'a> {
    py: Python<'a>,
    axes: &'a pyo3::types::PyAny,
}

impl<'a> Axes<'a> {
    pub fn scatter<I, J, F, G>(&self, x: I, y: J, alpha: f64) -> Result<&Self>
    where
        I: IntoIterator<Item = F>,
        J: IntoIterator<Item = G>,
        F: numpy::Element,
        G: numpy::Element,
    {
        let x: &PyArray1<F> = PyArray1::from_iter(self.py, x);
        let y: &PyArray1<G> = PyArray1::from_iter(self.py, y);
        self.axes.call_method(
            "plot",
            (x, y, "."),
            Some([("alpha", alpha), ("ms", 1.0)].into_py_dict(self.py)),
        )?;
        Ok(self)
    }

    pub fn line<I, J, F, G>(&self, x: I, y: J) -> Result<&Self>
    where
        I: IntoIterator<Item = F>,
        J: IntoIterator<Item = G>,
        F: numpy::Element,
        G: numpy::Element,
    {
        let x: &PyArray1<F> = PyArray1::from_iter(self.py, x);
        let y: &PyArray1<G> = PyArray1::from_iter(self.py, y);
        self.axes.call_method1("plot", (x, y))?;
        Ok(self)
    }

    pub fn show(&self) -> Result<&'a pyo3::PyAny> {
        Ok(self.axes.call_method0("show")?)
    }

    pub fn hist<I, F>(&self, x: I, bins: Option<usize>) -> Result<&Self>
    where
        I: IntoIterator<Item = F>,
        F: numpy::Element,
    {
        let x: &PyArray1<F> = PyArray1::from_iter(self.py, x);
        self.axes
            .call_method("hist", (x,), Some([("bins", bins)].into_py_dict(self.py)))?;
        Ok(self)
    }

    pub fn bar<I, F, J, G, K, H>(
        &self,
        x: I,
        height: J,
        widths: Option<K>,
        horizontal: bool,
    ) -> Result<&Self>
    where
        I: IntoIterator<Item = F>,
        J: IntoIterator<Item = G>,
        K: IntoIterator<Item = H>,
        F: numpy::Element,
        G: numpy::Element,
        H: numpy::Element,
    {
        let cmd = if horizontal { "barh" } else { "bar" };
        let bar_size = if horizontal { "height" } else { "width" };
        let x: &PyArray1<F> = PyArray1::from_iter(self.py, x);
        let h: &PyArray1<G> = PyArray1::from_iter(self.py, height);
        let widths: Option<&PyArray1<H>> =
            widths.map(|widths| PyArray1::from_iter(self.py, widths));
        self.axes.call_method(
            cmd,
            (x, h),
            widths.map(|widths| [(bar_size, widths)].into_py_dict(self.py)),
        )?;
        Ok(self)
    }

    pub fn heatmap<F>(&self, z: ndarray::ArrayView2<F>) -> Result<&Self>
    where
        F: numpy::Element,
    {
        let z = z.to_pyarray(self.py);
        self.axes.call_method1("imshow", (z,))?;
        Ok(self)
    }
}
