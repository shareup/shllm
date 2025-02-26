import Foundation

extension Bundle {
    // This is nearly identical to the generated code of `Bundle.module`. However,
    // `Bundle.module` does not work correctly in tests. There is a thread on the
    // Swift Forums describing the issue:
    // https://forums.swift.org/t/swift-5-3-spm-resources-in-tests-uses-wrong-bundle-path/37051
    static var shllm: Bundle {
        get throws {
            let bundleName = "SHLLM_SHLLM"

            var candidates = [
                // Bundle should be present here when the package is linked into an App.
                Bundle.main.resourceURL,

                // Bundle should be present here when the package is linked into a framework.
                Bundle(for: BundleLocator.self).resourceURL,

                // For command-line tools.
                Bundle.main.bundleURL,

                // iOS Xcode previews.
                Bundle(for: BundleLocator.self)
                    .resourceURL?
                    .deletingLastPathComponent()
                    .deletingLastPathComponent()
                    .deletingLastPathComponent(),

                // macOS Xcode previews.
                Bundle(for: BundleLocator.self)
                    .resourceURL?
                    .deletingLastPathComponent()
                    .deletingLastPathComponent(),
            ]

            // For tests
            // https://forums.swift.org/t/swift-5-3-spm-resources-in-tests-uses-wrong-bundle-path/37051/21
            candidates += Bundle.allBundles.compactMap(\.resourceURL)

            for candidate in candidates {
                let bundlePath = candidate?.appendingPathComponent(bundleName + ".bundle")
                if let bundle = bundlePath.flatMap(Bundle.init(url:)) {
                    return bundle
                }
            }

            throw SHLLMError.missingBundle(bundleName)
        }
    }

    func directory(named name: String) throws -> URL {
        if let url = url(forResource: name, withExtension: nil) {
            return url
        } else if let url = url(
            forResource: name,
            withExtension: nil,
            subdirectory: "Resources"
        ) {
            return url
        } else {
            throw SHLLMError.directoryNotFound(name)
        }
    }
}

private class BundleLocator: NSObject {}
