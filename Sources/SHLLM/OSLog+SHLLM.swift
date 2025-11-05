import Foundation
import os.log

let log = {
    let subsystem = Bundle.main.bundleIdentifier ?? "app.shareup.shllm"
    return OSLog(subsystem: subsystem, category: "shllm")
}()
