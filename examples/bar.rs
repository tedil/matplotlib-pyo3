use anyhow::Result;
use matplotlib_pyo3::PyPlot;

fn main() -> Result<()> {
    let x = vec![1, 3, 6, 10];
    let y = vec![4, 3, 2, 1];
    let widths = vec![1, 2, 3, 4];
    PyPlot::with_plt(|plt| {
        let fig = plt.figure()?;
        let ax = fig.gca()?;
        ax.bar(x, y, Some(widths), false)?;
        plt.show()?;
        Ok(())
    })?
}
