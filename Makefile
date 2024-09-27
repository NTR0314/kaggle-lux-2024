py-format:
	rye run ruff check python/rux_2024/ --select I --fix
	rye run ruff format python/rux_2024/
py-test:
	rye run pytest -vv --pyargs python/rux_2024/
py-lint:
	rye run ruff check python/rux_2024/
py-static:
	rye run mypy python/rux_2024/

r-format:
	cargo fmt
r-test:
	cargo test
r-lint:
	cargo clippy

build:
	maturin develop --skip-install

test: r-test py-test
check: r-lint py-lint py-static
prepare: build r-format py-format test check

clean:
	rm -rf .mypy_cache .pytest_cache .ruff_cache
