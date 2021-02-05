//! Like plot but using `matplotlib` as backend
use numpy::PyArray1;
use pyo3::types::IntoPyDict;
use pyo3::Python;
use std::iter::once;
use treelars::line::Segment;

use crate::io::Data;

#[derive(clap::Clap)]
pub(crate) struct PlotArgs {
    /// Take every `step`-th point into the plot
    #[clap(short, long, default_value = "500")]
    pub(crate) step: usize,

    /// Transparency of the points when plotting
    #[clap(short, long, default_value = "0.05")]
    pub(crate) alpha: f64,

    #[clap(short, long, default_value = "80")]
    pub(crate) max_count: u8,
}

/// Plot the results of `lasso` using Matplotlib
#[derive(clap::Clap)]
pub struct Args {
    #[clap(parse(from_os_str))]
    counts: std::path::PathBuf,

    #[clap(short = 'S', long, parse(from_os_str))]
    segments: Option<std::path::PathBuf>,

    #[clap(short, long)]
    target: Option<String>,

    #[clap(long)]
    dump: bool,

    #[clap(flatten)]
    plot: PlotArgs,
}

pub fn main(args: Args) -> crate::Result {
    use rand::prelude::Rng;

    let data = args
        .segments
        .clone()
        .map(|s| Data::read_file(&s))
        .transpose()?;
    let target: String = data
        .map(|d| d.target)
        .or_else(|| args.target.clone())
        .ok_or("Need --segments or --target")?;
    let mut countfile = crate::counts::CountFile::from_path(&args.counts)?;
    let (counts, positions) = countfile.get(&target)?.to_vecs();
    let coverage = countfile.coverage()?;
    let segments = args
        .segments
        .as_ref()
        .map(crate::io::Data::read_file)
        .transpose()?;

    Python::with_gil(|py| {
        let plt = PyPlot::new(py)?;
        let fig = plt.figure()?;
        let ax = fig.gca()?;
        {
            let x = positions.iter().map(|&p| p as f32).step_by(args.plot.step);
            let mut rng = rand::thread_rng();
            let y = counts
                .iter()
                .map(|v| v.min(&args.plot.max_count))
                .map(|&v| v as f32)
                .map(move |v| v + rng.gen::<f32>() - 0.5)
                .map(|v| (v * 2.0 / coverage as f32))
                .step_by(args.plot.step);
            ax.scatter(x, y, args.plot.alpha)?;
        }
        segments
            .map(|s| {
                let (x, y) = segments_to_points(&s.segments, &positions);
                if args.dump {
                    let (x, y) = (x.clone(), y.clone());
                    assert_eq!(x.clone().count(), y.clone().count());
                    for (xi, yi) in x.zip(y) {
                        use thousands::Separable;
                        println!("{:>12}, {:.3}", xi.separate_with_underscores(), yi);
                    }
                }
                ax.line(x, y)
            })
            .transpose()?;
        plt.show()?;
        Ok(())
    })
}

pub fn segments_to_points<'a>(
    segments: &'a [Segment<f64>],
    positions: &'a [u32],
) -> (
    impl Iterator<Item = u32> + 'a + Clone,
    impl Iterator<Item = f64> + 'a + Clone,
) {
    let x = segments.iter().flat_map(move |(r, _)| {
        once(positions[r.start]).chain(itertools::repeat_n(positions[r.end - 1], 2))
    });
    let y = segments
        .iter()
        .flat_map(|(_, v)| itertools::repeat_n(v, 2).chain(once(&std::f64::NAN)))
        .cloned();
    (x, y)
}

pub struct PyPlot<'a> {
    py: Python<'a>,
    plt: &'a pyo3::types::PyModule,
}

impl<'a> PyPlot<'a> {
    pub fn new(py: Python<'a>) -> crate::Result<Self> {
        let plt = py.import("matplotlib.pyplot")?;
        Ok(Self { py, plt })
    }

    pub fn figure(&self) -> crate::Result<Figure> {
        let fig = self.plt.call0("figure")?;
        Ok(Figure { py: self.py, fig })
    }

    pub fn show(&self) -> crate::Result<&'a pyo3::PyAny> {
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
    ) -> crate::Result<Axes> {
        let args = PyArray1::from_vec(self.py, vec![left, bottom, width, height]);
        let mut shares = vec![];
        if let Some(ax) = share_x {
            shares.push(("sharex", ax.axes));
        }
        if let Some(ax) = share_y {
            shares.push(("sharey", ax.axes));
        }
        let axis = self.fig.call_method(
            "add_axes",
            (args,),
            Some(shares.into_py_dict(self.py)),
        )?;
        Ok(Axes {
            py: self.py,
            fig: &self,
            axes: axis,
        })
    }

    pub fn gca(&self) -> crate::Result<Axes> {
        let axes = self.fig.call_method0("gca")?;
        Ok(Axes {
            py: self.py,
            fig: &self,
            axes,
        })
    }

    pub fn show(&self) -> crate::Result<&'a pyo3::PyAny> {
        Ok(self.fig.call_method0("show")?)
    }
}

pub struct Axes<'a> {
    py: Python<'a>,
    #[allow(dead_code)]
    fig: &'a Figure<'a>,
    axes: &'a pyo3::types::PyAny,
}

impl<'a> Axes<'a> {
    pub fn scatter<I, J, F, G>(&self, x: I, y: J, alpha: f64) -> crate::Result<&Self>
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

    pub fn line<I, J, F, G>(&self, x: I, y: J) -> crate::Result<&Self>
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

    pub fn show(&self) -> crate::Result<&'a pyo3::PyAny> {
        Ok(self.axes.call_method0("show")?)
    }

    pub fn hist<I, F>(&self, x: I, bins: Option<usize>) -> crate::Result<&Self>
    where
        I: IntoIterator<Item = F>,
        F: numpy::Element,
    {
        let x: &PyArray1<F> = PyArray1::from_iter(self.py, x);
        self.axes.call_method(
            "hist",
            (x,),
            Some([("bins", bins)].into_py_dict(self.py)),
        )?;
        Ok(self)
    }

    pub fn bar<I, F, J, G>(
        &self,
        x: I,
        height: J,
        horizontal: bool,
    ) -> crate::Result<&Self>
    where
        I: IntoIterator<Item = F>,
        J: IntoIterator<Item = G>,
        F: numpy::Element,
        G: numpy::Element,
    {
        let cmd = if horizontal { "barh" } else { "bar" };
        let x: &PyArray1<F> = PyArray1::from_iter(self.py, x);
        let h: &PyArray1<G> = PyArray1::from_iter(self.py, height);
        self.axes.call_method1(cmd, (x, h))?;
        Ok(self)
    }
}
