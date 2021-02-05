use anyhow::Result;
use matplotlib_pyo3::*;
use ndarray::array;
use pyo3::Python;

fn main() -> Result<()> {
    let data = array![
        [0.1, 0.2, 0.3, 0.4],
        [0.2, 0.3, 0.4, 0.1],
        [0.3, 0.4, 0.1, 0.2],
        [0.4, 0.1, 0.2, 0.3]
    ];
    Python::with_gil(|py| {
        let plt = PyPlot::new(py)?;
        let fig = plt.figure()?;
        let ax = fig.gca()?;
        ax.heatmap(data.view())?;
        plt.show()?;
        Ok(())
    })
}
