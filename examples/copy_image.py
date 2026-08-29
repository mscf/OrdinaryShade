import ordinaryshade as osh


@osh.compute(workgroup_size=(8, 8, 1))
def copy_and_dim(
    source: osh.storage_image("rgba16f", access="read"),
    target: osh.storage_image("rgba16f", access="write"),
):
    pixel = osh.global_invocation_id.xy
    color = source.load(pixel)
    target.store(pixel, color * 0.5)


if __name__ == "__main__":
    result = osh.compile(copy_and_dim)
    print(result.source)

